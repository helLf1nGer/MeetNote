import logging
from typing import List, Dict
from .groq_api_helper import groq_api_call, count_tokens
from .rate_limiter import RateLimiter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GroqLLMCombiner:
    def __init__(self, model="llama3-groq-70b-8192-tool-use-preview"):
        self.model = model
        self.max_tokens = 7500  # Leave some room for the prompt and response

    @RateLimiter(max_calls=30, period=60)
    def combine(self, transcription: List[Dict]) -> List[Dict]:
        chunks = self._split_transcription(transcription)
        logger.info(f"Split transcription into {len(chunks)} chunks")

        combined_results = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            prompt = self._prepare_prompt(chunk)
            prompt_tokens = count_tokens(prompt)
            
            if prompt_tokens > self.max_tokens:
                logger.warning(f"Chunk {i+1} exceeds token limit. Splitting further.")
                sub_chunks = self._split_chunk(chunk, prompt_tokens)
                for j, sub_chunk in enumerate(sub_chunks):
                    logger.info(f"Processing sub-chunk {j+1}/{len(sub_chunks)} of chunk {i+1}")
                    sub_prompt = self._prepare_prompt(sub_chunk)
                    response = self._generate_response(sub_prompt)
                    combined_results.extend(self._parse_response(response, sub_chunk))
            else:
                response = self._generate_response(prompt)
                combined_results.extend(self._parse_response(response, chunk))

        return combined_results

    def _split_transcription(self, transcription: List[Dict]) -> List[List[Dict]]:
        chunks = []
        current_chunk = []
        current_tokens = 0

        for segment in transcription:
            segment_tokens = count_tokens(segment['text'])
            if current_tokens + segment_tokens > self.max_tokens and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0
            
            current_chunk.append(segment)
            current_tokens += segment_tokens

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _split_chunk(self, chunk: List[Dict], prompt_tokens: int) -> List[List[Dict]]:
        sub_chunks = []
        current_sub_chunk = []
        current_tokens = 0
        target_tokens = self.max_tokens // 2  # Aim for half the max tokens to ensure we stay under the limit

        for segment in chunk:
            segment_tokens = count_tokens(segment['text'])
            if current_tokens + segment_tokens > target_tokens and current_sub_chunk:
                sub_chunks.append(current_sub_chunk)
                current_sub_chunk = []
                current_tokens = 0
            
            current_sub_chunk.append(segment)
            current_tokens += segment_tokens

        if current_sub_chunk:
            sub_chunks.append(current_sub_chunk)

        return sub_chunks

    def _prepare_prompt(self, chunk: List[Dict]) -> str:
        prompt = "Analyze the following transcript and determine the most likely speaker for each segment. Assign speakers as SPEAKER_00, SPEAKER_01, etc. Consider the context and speaking style when making assignments. Provide your analysis as a JSON array of segments. Each segment should have the following structure:\n"
        prompt += "{\n  \"start\": float,\n  \"end\": float,\n  \"text\": string,\n  \"speaker\": string\n}\n\n"
        prompt += "Transcript:\n"
        for segment in chunk:
            prompt += f"[{segment['start']} - {segment['end']}]: {segment['text']}\n"
        prompt += "\nJSON response:"
        return prompt

    def _generate_response(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = groq_api_call(messages, self.model, max_tokens=self.max_tokens)
        return response

    def _parse_response(self, response: str, chunk: List[Dict]) -> List[Dict]:
        try:
            parsed_response = eval(response)  # Using eval as the response is already in Python list format
            if not isinstance(parsed_response, list):
                raise ValueError("Response is not a list")
            
            for segment in parsed_response:
                if not all(key in segment for key in ['start', 'end', 'text', 'speaker']):
                    raise ValueError(f"Invalid segment structure: {segment}")
            
            logger.info(f"Successfully parsed {len(parsed_response)} segments")
            return parsed_response
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            logger.error(f"Raw response: {response}")
            # Fallback: return the original chunk with default speaker assignments
            return [{'start': seg['start'], 'end': seg['end'], 'text': seg['text'], 'speaker': 'SPEAKER_00'} for seg in chunk]

def combine(transcription: List[Dict], diarization: List[Dict]) -> List[Dict]:
    groq_combiner = GroqLLMCombiner()
    return groq_combiner.combine(transcription)