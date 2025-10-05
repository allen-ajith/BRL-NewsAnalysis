import pandas as pd
import google.generativeai as genai
import logging
import time
import os
import argparse
import json
import re
from typing import Optional, Dict, List
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class NewsProcessor:
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.setup_logging()
        self.checkpoint_lock = threading.Lock()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('inference_paid.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_prompt(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()

    def load_data(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        if 'id' not in df.columns:
            df['id'] = range(len(df))
        df['HEADING'] = df['HEADING'].fillna('')
        df['ARTICLE CONTENT'] = df['ARTICLE CONTENT'].fillna('')
        self.logger.info(f"Loaded {len(df)} records from {file_path}")
        return df

    def save_checkpoint(self, df: pd.DataFrame, file_path: str):
        with self.checkpoint_lock:
            temp_file = f"{file_path}.tmp"
            df.to_csv(temp_file, index=False)
            os.replace(temp_file, file_path)

    def load_checkpoint(self, file_path: str) -> Optional[pd.DataFrame]:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            self.logger.info(f"Resumed from checkpoint with {len(df)} rows: {file_path}")
            return df
        return None

    def call_api_with_retries(self, prompt: str, temperature: float, top_p: float, top_k: int, max_attempts: int = 5) -> str:
        for attempt in range(max_attempts):
            try:
                generation_config = genai.types.GenerationConfig(
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_output_tokens=8192
                )
                safety_settings = [
                    {"category": f"HARM_CATEGORY_{cat}", "threshold": "BLOCK_NONE"}
                    for cat in ["HARASSMENT", "HATE_SPEECH", "SEXUALLY_EXPLICIT", "DANGEROUS_CONTENT"]
                ]
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )

                if not response.text:
                    finish_reason = response.candidates[0].finish_reason if response.candidates else "unknown"
                    return f"API_BLOCKED: {finish_reason}"
                
                return response.text.strip()

            except Exception as e:
                error_msg = str(e).lower()
                is_retryable = any(term in error_msg for term in 
                    ["500", "503", "unavailable", "resource_exhausted", "quota", "rate_limit", "429"])
                
                if is_retryable and attempt < max_attempts - 1:
                    delay = min((2 ** attempt) + 1, 60)  # Cap delay at 60 seconds
                    self.logger.warning(f"Retryable API error encountered. Retrying in {delay}s... (Attempt {attempt + 1}/{max_attempts})")
                    time.sleep(delay)
                else:
                    self.logger.error(f"Non-retryable error or max retries reached: {e}")
                    return f"API_ERROR: {e}"
        return "API_ERROR: Max retries exceeded"

    def parse_batch_response(self, response_text: str, batch_ids: List[int]) -> Dict:
        try:
            # Try to find JSON array with or without code blocks
            json_match = re.search(r'```json\s*(\[.*?\])\s*```', response_text, re.DOTALL)
            if not json_match:
                json_match = re.search(r'(\[.*?\])', response_text, re.DOTALL)
            
            if not json_match:
                # Try to find partial JSON and reconstruct
                objects = re.findall(r'\{\s*"id":\s*\d+.*?\}', response_text, re.DOTALL)
                if objects:
                    clean_json = '[' + ','.join(objects) + ']'
                else:
                    raise json.JSONDecodeError("No JSON array or objects found in the response.", response_text, 0)
            else:
                clean_json = json_match.group(1)
            
            results = json.loads(clean_json)
            
            # Ensure all results have required fields
            processed_results = {}
            for result in results:
                if 'id' in result:
                    processed_results[result['id']] = {
                        'sentiment': result.get('sentiment', 'PARSE_ERROR'),
                        'reasoning': result.get('reasoning', 'Missing reasoning'),
                        'raw': str(result)
                    }
            
            return processed_results
        
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"Failed to parse JSON response: {e}\nRaw response:\n{response_text[:1000]}...")
            return {
                batch_id: {
                    'sentiment': 'PARSE_ERROR',
                    'reasoning': f'JSON parsing failed: {e}',
                    'raw': response_text[:500]
                } for batch_id in batch_ids
            }

    def process_batch(self, prompt_template: str, batch_df: pd.DataFrame, temperature: float, top_p: float, top_k: int, rate_limit_delay: float) -> Dict:
        articles_to_process = batch_df[['id', 'HEADING', 'ARTICLE CONTENT']].to_dict(orient='records')
        articles_json_str = json.dumps(articles_to_process, indent=2)
        full_prompt = prompt_template.replace("{articles_json}", articles_json_str)

        # Add rate limiting delay
        time.sleep(rate_limit_delay)
        
        response_text = self.call_api_with_retries(full_prompt, temperature, top_p, top_k)
        
        if response_text.startswith("API_"):
            return {
                row['id']: {
                    'sentiment': 'API_ERROR', 'reasoning': response_text, 'raw': response_text
                } for _, row in batch_df.iterrows()
            }
            
        batch_ids = batch_df['id'].tolist()
        return self.parse_batch_response(response_text, batch_ids)

    def process(self, csv_file: str, prompt_file: str, output_file: str,
                checkpoint_file: str, max_rows: int,
                temperature: float, top_p: float, top_k: int,
                batch_size: int, workers: int, rate_limit_delay: float, checkpoint_frequency: int):
        
        prompt_template = self.load_prompt(prompt_file)
        
        df = self.load_checkpoint(checkpoint_file)
        if df is None:
            df = self.load_data(csv_file)
            if max_rows:
                df = df.head(max_rows).copy()
            df['pred_sentiment'] = None
            df['pred_reasoning'] = None
            df['pred_raw'] = None
            
        to_process_df = df[df['pred_sentiment'].isna()].copy()
        
        if to_process_df.empty:
            self.logger.info("All records have already been processed.")
            df.to_csv(output_file, index=False)
            return

        total_to_process = len(to_process_df)
        self.logger.info(f"Processing {total_to_process} remaining articles in batches of {batch_size}...")
        
        batches = [to_process_df.iloc[i:i + batch_size] for i in range(0, total_to_process, batch_size)]
        total_batches = len(batches)
        processed_batches = 0
        
        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_batch_df = {
                    executor.submit(self.process_batch, prompt_template, batch, temperature, top_p, top_k, rate_limit_delay): batch 
                    for batch in batches
                }
                
                for future in as_completed(future_to_batch_df):
                    try:
                        results_dict = future.result()
                        batch_df = future_to_batch_df[future]
                        
                        # Update main DataFrame
                        for article_id, result_data in results_dict.items():
                            idx = df.index[df['id'] == article_id].tolist()
                            if idx:
                                df.loc[idx[0], 'pred_sentiment'] = result_data.get('sentiment', 'ERROR')
                                df.loc[idx[0], 'pred_reasoning'] = result_data.get('reasoning', 'ERROR')
                                df.loc[idx[0], 'pred_raw'] = str(result_data)
                        
                        processed_batches += 1
                        success_count = sum(1 for r in results_dict.values() if not str(r['sentiment']).startswith(('API_', 'PARSE_')))
                        
                        self.logger.info(f"Progress: {processed_batches}/{total_batches} batches | "
                                       f"Success: {success_count}/{len(batch_df)} articles in this batch")
                        
                        # Save checkpoint every N batches or if it's the last batch
                        if processed_batches % checkpoint_frequency == 0 or processed_batches == total_batches:
                            self.save_checkpoint(df, checkpoint_file)
                            self.logger.info(f"Checkpoint saved at batch {processed_batches}")
                            
                    except Exception as exc:
                        self.logger.error(f"A batch generated an exception: {exc}")

        except KeyboardInterrupt:
            self.logger.info("Process interrupted by user. Saving checkpoint...")
        finally:
            self.save_checkpoint(df, checkpoint_file)
            df.to_csv(output_file, index=False)
            completed = len(df[df['pred_sentiment'].notna()])
            self.logger.info(f"Processing finished. {completed}/{len(df)} articles completed. Final results saved to {output_file}")


def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description='Process news with Gemini (Optimized Batching Version)')
    parser.add_argument('--csv', required=True, help='Input CSV file')
    parser.add_argument('--prompt', required=True, help='Batch-compatible prompt file')
    parser.add_argument('--output', required=True, help='Output CSV file')
    parser.add_argument('--checkpoint', default='checkpoint.csv', help='Checkpoint file name')
    parser.add_argument('--api-key', help='API key (or set GEMINI_API_KEY in .env)')
    parser.add_argument('--model', default='gemini-1.5-flash', help='Model name to use')
    parser.add_argument('--max-rows', type=int, help='Limit total number of rows to process')
    parser.add_argument('--reset', action='store_true', help='Delete checkpoint file to start from scratch')
    parser.add_argument('--temperature', type=float, default=0.1, help='LLM temperature')
    parser.add_argument('--top-p', type=float, default=1.0, help='LLM top_p')
    parser.add_argument('--top-k', type=int, default=None, help='LLM top_k')
    parser.add_argument('--batch-size', type=int, default=25, help='Number of articles per API call (25-180 recommended based on article length)')
    parser.add_argument('--workers', type=int, default=3, help='Number of parallel API calls (2-5 recommended)')
    parser.add_argument('--rate-limit-delay', type=float, default=0.3, help='Delay between API calls in seconds')
    parser.add_argument('--checkpoint-frequency', type=int, default=10, help='Save checkpoint every N batches')
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("API key not found. Please provide it via --api-key or set GEMINI_API_KEY in a .env file.")
    
    if args.reset and os.path.exists(args.checkpoint):
        os.remove(args.checkpoint)
        print(f"Removed checkpoint file: {args.checkpoint}")
        
    processor = NewsProcessor(api_key, args.model)
    
    processor.process(
        csv_file=args.csv,
        prompt_file=args.prompt,
        output_file=args.output,
        checkpoint_file=args.checkpoint,
        max_rows=args.max_rows,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        batch_size=args.batch_size,
        workers=args.workers,
        rate_limit_delay=args.rate_limit_delay,
        checkpoint_frequency=args.checkpoint_frequency
    )

if __name__ == "__main__":
    main()