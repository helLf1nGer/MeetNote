import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalLlamaTinyCombiner:
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
        self.max_tokens = 2000

    def combine(self, semantic_flow_output: List[Dict], diarization: List[Dict]) -> List[Dict]:
        problematic_chunks = self._find_problematic_chunks(semantic_flow_output, diarization)
        logger.info(f"Found {len(problematic_chunks)} problematic chunks:")
        for i, chunk in enumerate(problematic_chunks):
            logger.info(f"Chunk {i+1}:")
            logger.info(f"  Text: {chunk['text']}")
            logger.info(f"  Possible speakers: {chunk['possible_speakers']}")
        
        if problematic_chunks:
            resolved_chunks = self._resolve_problematic_chunks(problematic_chunks, semantic_flow_output)
            return self._merge_results(semantic_flow_output, resolved_chunks)
        return semantic_flow_output

    def _find_problematic_chunks(self, semantic_flow_output: List[Dict], diarization: List[Dict]) -> List[Dict]:
        problematic_chunks = []
        for i, segment in enumerate(semantic_flow_output):
            segment_speakers = self._get_speakers_for_segment(segment, diarization)
            if len(segment_speakers) > 1:
                problematic_chunks.append({
                    **segment,
                    'possible_speakers': list(segment_speakers),
                    'index': i
                })
        return problematic_chunks

    def _get_speakers_for_segment(self, segment: Dict, diarization: List[Dict]) -> set:
        start, end = segment['start'], segment['end']
        return set(d['speaker'] for d in diarization if d['start'] < end and d['end'] > start)

    def _resolve_problematic_chunks(self, problematic_chunks: List[Dict], semantic_flow_output: List[Dict]) -> List[Dict]:
        augmented_chunks = self._augment_chunks(problematic_chunks, semantic_flow_output)
        tokenized_chunks = self._tokenize_chunks(augmented_chunks)
        split_chunks = self._split_large_chunks(tokenized_chunks)
        
        resolved_chunks = []
        for chunk_group in split_chunks:
            prompt = self._prepare_prompt(chunk_group)
            logger.info("Input to TinyLlama:")
            logger.info(prompt)
            
            response = self._generate_response(prompt)
            logger.info("Output from TinyLlama:")
            logger.info(response)
            
            resolved_chunks.extend(self._parse_response(response, chunk_group))
        
        return resolved_chunks

    def _augment_chunks(self, problematic_chunks: List[Dict], semantic_flow_output: List[Dict]) -> List[Dict]:
        augmented_chunks = []
        for chunk in problematic_chunks:
            index = chunk['index']
            prev_text = semantic_flow_output[index - 1]['text'] if index > 0 else ""
            next_text = semantic_flow_output[index + 1]['text'] if index < len(semantic_flow_output) - 1 else ""
            augmented_chunks.append({
                **chunk,
                'prev_text': prev_text,
                'next_text': next_text
            })
        return augmented_chunks

    def _tokenize_chunks(self, chunks: List[Dict]) -> List[Dict]:
        for chunk in chunks:
            chunk['tokens'] = len(self.tokenizer.encode(chunk['prev_text'] + chunk['text'] + chunk['next_text']))
        return chunks

    def _split_large_chunks(self, chunks: List[Dict]) -> List[List[Dict]]:
        split_chunks = []
        current_group = []
        current_tokens = 0
        
        for chunk in chunks:
            if current_tokens + chunk['tokens'] > self.max_tokens and current_group:
                split_chunks.append(current_group)
                current_group = []
                current_tokens = 0
            
            current_group.append(chunk)
            current_tokens += chunk['tokens']
        
        if current_group:
            split_chunks.append(current_group)
        
        return split_chunks

    def _prepare_prompt(self, chunk_group: List[Dict]) -> str:
        prompt = "Determine the most likely speaker for each segment based on semantic continuity. Choose only one speaker per segment. Respond with ONLY the chosen speaker for each segment, separated by newlines. Use the format 'Segment X: SPEAKER_XX' where XX is the speaker number.\n\n"
        for i, chunk in enumerate(chunk_group):
            prompt += f"Segment {i+1}:\n"
            prompt += f"Previous text: {chunk['prev_text']}\n"
            prompt += f"Current text: {chunk['text']}\n"
            prompt += f"Next text: {chunk['next_text']}\n"
            prompt += f"Possible speakers: {', '.join(chunk['possible_speakers'])}\n\n"
        return prompt

    def _generate_response(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_tokens).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=500, num_return_sequences=1, temperature=0.0, do_sample=False)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Raw response from TinyLlama:\n{response}")
        return response

    def _parse_response(self, response: str, chunk_group: List[Dict]) -> List[Dict]:
        resolved_chunks = []
        lines = response.strip().split('\n')
        segment_pattern = re.compile(r'Segment (\d+): (SPEAKER_\d+)')
        
        for chunk in chunk_group:
            speaker = chunk['possible_speakers'][0]  # Default to first possible speaker
            
            for line in lines:
                match = segment_pattern.search(line)
                if match:
                    segment_num = int(match.group(1))
                    if segment_num == len(resolved_chunks) + 1:
                        proposed_speaker = match.group(2)
                        if proposed_speaker in chunk['possible_speakers']:
                            speaker = proposed_speaker
                        else:
                            logger.warning(f"Invalid speaker choice: {proposed_speaker}. Using {speaker}.")
                        break
            
            resolved_chunks.append({**chunk, 'speaker': speaker})
            logger.info(f"Assigned speaker {speaker} to segment {len(resolved_chunks)}")
        
        return resolved_chunks

    def _merge_results(self, original_output: List[Dict], resolved_chunks: List[Dict]) -> List[Dict]:
        resolved_indices = {chunk['index'] for chunk in resolved_chunks}
        final_output = []
        for i, segment in enumerate(original_output):
            if i in resolved_indices:
                resolved_chunk = next(c for c in resolved_chunks if c['index'] == i)
                final_output.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text'],
                    'speaker': resolved_chunk['speaker']
                })
            else:
                final_output.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text'],
                    'speaker': segment['speaker']
                })
        return final_output

def combine(transcription: List[Dict], diarization: List[Dict]) -> List[Dict]:
    from .semantic_flow_combiner import SemanticFlowCombiner
    semantic_flow = SemanticFlowCombiner()
    semantic_flow_output = semantic_flow.combine(transcription, diarization)
    
    llama_combiner = LocalLlamaTinyCombiner()
    segments = llama_combiner.combine(semantic_flow_output, diarization)
    
    return segments