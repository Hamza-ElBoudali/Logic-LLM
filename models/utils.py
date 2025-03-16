import backoff  # for exponential backoff
import openai
import os
import asyncio
import requests
from typing import Any, List, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define RateLimitError for compatibility with both old and new OpenAI versions
try:
    # Try to get RateLimitError from the new location (openai.RateLimitError)
    RateLimitError = openai.RateLimitError
except AttributeError:
    try:
        # Try to get RateLimitError from the old location (openai.error.RateLimitError)
        RateLimitError = openai.error.RateLimitError
    except AttributeError:
        # If neither works, create a generic exception class
        class RateLimitError(Exception):
            pass

# Check if we're using the new OpenAI client
USING_NEW_OPENAI_CLIENT = hasattr(openai, "OpenAI")

@backoff.on_exception(backoff.expo, RateLimitError)
def completions_with_backoff(**kwargs):
    if USING_NEW_OPENAI_CLIENT:
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return client.completions.create(**kwargs)
    else:
        return openai.Completion.create(**kwargs)

@backoff.on_exception(backoff.expo, RateLimitError)
def chat_completions_with_backoff(**kwargs):
    if USING_NEW_OPENAI_CLIENT:
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return client.chat.completions.create(**kwargs)
    else:
        return openai.ChatCompletion.create(**kwargs)

async def dispatch_openai_chat_requests(
    messages_list: list[list[dict[str,Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop_words: list[str]
) -> list[str]:
    """Dispatches requests to OpenAI API asynchronously.
    
    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.
        stop_words: List of words to stop the model from generating.
    Returns:
        List of responses from OpenAI API.
    """
    if USING_NEW_OPENAI_CLIENT:
        # For new OpenAI client, we need to use a different approach
        client = openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        async_responses = []
        for messages in messages_list:
            async_responses.append(
                client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stop=stop_words
                )
            )
    else:
        # For old OpenAI client
        async_responses = [
            openai.ChatCompletion.acreate(
                model=model,
                messages=x,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop_words
            )
            for x in messages_list
        ]
    
    responses = await asyncio.gather(*async_responses)
    
    # Extract content based on client version
    if USING_NEW_OPENAI_CLIENT:
        return [response.choices[0].message.content for response in responses]
    else:
        return [response["choices"][0]["message"]["content"] for response in responses]

async def dispatch_openai_prompt_requests(
    messages_list: list[list[dict[str,Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop_words: list[str]
) -> list[str]:
    if USING_NEW_OPENAI_CLIENT:
        # For new OpenAI client, we need to use a different approach
        client = openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        async_responses = []
        for prompt in messages_list:
            async_responses.append(
                client.completions.create(
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stop=stop_words
                )
            )
    else:
        # For old OpenAI client
        async_responses = [
            openai.Completion.acreate(
                model=model,
                prompt=x,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop_words
            )
            for x in messages_list
        ]
    
    responses = await asyncio.gather(*async_responses)
    
    # Extract content based on client version
    if USING_NEW_OPENAI_CLIENT:
        return [response.choices[0].text for response in responses]
    else:
        return [response["choices"][0]["text"] for response in responses]

class OpenAIModel:
    def __init__(self, API_KEY, model_name, stop_words, max_new_tokens) -> None:
        self.api_key = API_KEY
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.stop_words = stop_words
        
        # Set up the client based on the OpenAI version
        if USING_NEW_OPENAI_CLIENT:
            self.client = openai.OpenAI(api_key=API_KEY)
        else:
            openai.api_key = API_KEY

    # used for chat-gpt and gpt-4
    def chat_generate(self, input_string, temperature = 0.0):
        if USING_NEW_OPENAI_CLIENT:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": input_string}
                ],
                max_tokens=self.max_new_tokens,
                temperature=temperature,
                top_p=1.0,
                stop=self.stop_words
            )
            generated_text = response.choices[0].message.content.strip()
        else:
            response = chat_completions_with_backoff(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": input_string}
                ],
                max_tokens=self.max_new_tokens,
                temperature=temperature,
                top_p=1.0,
                stop=self.stop_words
            )
            generated_text = response['choices'][0]['message']['content'].strip()
        
        return generated_text
    
    # used for text/code-davinci
    def prompt_generate(self, input_string, temperature = 0.0):
        if USING_NEW_OPENAI_CLIENT:
            response = self.client.completions.create(
                model=self.model_name,
                prompt=input_string,
                max_tokens=self.max_new_tokens,
                temperature=temperature,
                top_p=1.0,
                stop=self.stop_words
            )
            generated_text = response.choices[0].text.strip()
        else:
            response = completions_with_backoff(
                model=self.model_name,
                prompt=input_string,
                max_tokens=self.max_new_tokens,
                temperature=temperature,
                top_p=1.0,
                stop=self.stop_words
            )
            generated_text = response['choices'][0]['text'].strip()
        
        return generated_text

    def generate(self, input_string, temperature = 0.0):
        if self.model_name in ['text-davinci-002', 'code-davinci-002', 'text-davinci-003']:
            return self.prompt_generate(input_string, temperature)
        elif self.model_name in ['gpt-4', 'gpt-3.5-turbo', 'gpt-4-turbo', 'gpt-3.5-turbo-16k']:
            return self.chat_generate(input_string, temperature)
        else:
            # Default to chat model for newer models
            return self.chat_generate(input_string, temperature)
    
    def batch_chat_generate(self, messages_list, temperature = 0.0):
        open_ai_messages_list = []
        for message in messages_list:
            open_ai_messages_list.append(
                [{"role": "user", "content": message}]
            )
        
        predictions = asyncio.run(
            dispatch_openai_chat_requests(
                open_ai_messages_list, self.model_name, temperature, self.max_new_tokens, 1.0, self.stop_words
            )
        )
        
        return predictions  # The dispatch function now handles the response format differences
    
    def batch_prompt_generate(self, prompt_list, temperature = 0.0):
        predictions = asyncio.run(
            dispatch_openai_prompt_requests(
                prompt_list, self.model_name, temperature, self.max_new_tokens, 1.0, self.stop_words
            )
        )
        
        return predictions  # The dispatch function now handles the response format differences

    def batch_generate(self, messages_list, temperature = 0.0):
        if self.model_name in ['text-davinci-002', 'code-davinci-002', 'text-davinci-003']:
            return self.batch_prompt_generate(messages_list, temperature)
        elif self.model_name in ['gpt-4', 'gpt-3.5-turbo', 'gpt-4-turbo', 'gpt-3.5-turbo-16k']:
            return self.batch_chat_generate(messages_list, temperature)
        else:
            # Default to chat model for newer models
            return self.batch_chat_generate(messages_list, temperature)

    def generate_insertion(self, input_string, suffix, temperature = 0.0):
        if USING_NEW_OPENAI_CLIENT:
            response = self.client.completions.create(
                model=self.model_name,
                prompt=input_string,
                suffix=suffix,
                max_tokens=self.max_new_tokens,
                temperature=temperature,
                top_p=1.0,
                stop=self.stop_words
            )
            generated_text = response.choices[0].text.strip()
        else:
            response = completions_with_backoff(
                model=self.model_name,
                prompt=input_string,
                suffix=suffix,
                max_tokens=self.max_new_tokens,
                temperature=temperature,
                top_p=1.0,
                stop=self.stop_words
            )
            generated_text = response['choices'][0]['text'].strip()
        
        return generated_text

class OpenRouterError(Exception):
    """Exception raised for OpenRouter API errors."""
    pass

@backoff.on_exception(backoff.expo, OpenRouterError)
def openrouter_chat_completions_with_backoff(**kwargs):
    """Make a request to OpenRouter API with exponential backoff."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {kwargs.pop('api_key')}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Logic-LLM",  # Optional
        "X-Title": "Logic-LLM Framework"  # Optional
    }
    
    # Extract timeout if provided
    timeout = kwargs.pop('timeout', 60)  # Default 60 seconds timeout
    
    # Print debug info if verbose
    verbose = kwargs.pop('verbose', False)
    if verbose:
        print(f"Making OpenRouter API request with timeout={timeout}s")
        print(f"Request data: {kwargs}")
    
    try:
        response = requests.post(url, headers=headers, json=kwargs, timeout=timeout)
        
        if response.status_code != 200:
            error_msg = f"OpenRouter API error: {response.status_code} - {response.text}"
            if verbose:
                print(error_msg)
            raise OpenRouterError(error_msg)
        
        if verbose:
            print(f"OpenRouter API request successful")
        
        return response.json()
    except requests.exceptions.Timeout:
        error_msg = f"OpenRouter API request timed out after {timeout} seconds"
        if verbose:
            print(error_msg)
        raise OpenRouterError(error_msg)
    except Exception as e:
        error_msg = f"OpenRouter API request failed: {str(e)}"
        if verbose:
            print(error_msg)
        raise OpenRouterError(error_msg)

async def dispatch_openrouter_chat_requests(
    messages_list: List[List[Dict[str, Any]]],
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop_words: List[str],
    timeout: int = 60,
    verbose: bool = False
) -> List[Dict]:
    """Dispatches requests to OpenRouter API asynchronously."""
    async def make_request(messages):
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Logic-LLM",  # Optional
            "X-Title": "Logic-LLM Framework"  # Optional
        }
        
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stop": stop_words
        }
        
        if verbose:
            print(f"Making async OpenRouter API request with timeout={timeout}s")
            print(f"Request data: {data}")
        
        try:
            session = requests.Session()
            response = await asyncio.to_thread(
                session.post, url, headers=headers, json=data, timeout=timeout
            )
            
            if response.status_code != 200:
                error_msg = f"OpenRouter API error: {response.status_code} - {response.text}"
                if verbose:
                    print(error_msg)
                raise OpenRouterError(error_msg)
            
            if verbose:
                print(f"Async OpenRouter API request successful")
            
            return response.json()
        except requests.exceptions.Timeout:
            error_msg = f"Async OpenRouter API request timed out after {timeout} seconds"
            if verbose:
                print(error_msg)
            raise OpenRouterError(error_msg)
        except Exception as e:
            error_msg = f"Async OpenRouter API request failed: {str(e)}"
            if verbose:
                print(error_msg)
            raise OpenRouterError(error_msg)
    
    tasks = [make_request(messages) for messages in messages_list]
    return await asyncio.gather(*tasks)

class DeepSeekModel:
    """Class for interacting with DeepSeek models via OpenRouter API."""
    
    def __init__(self, API_KEY, model_name="deepseek/deepseek-r1:free", stop_words=None, max_new_tokens=1024, timeout=60, verbose=False) -> None:
        self.api_key = API_KEY
        self.model_name = model_name
        self.stop_words = stop_words if stop_words else []
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout
        self.verbose = verbose
    
    def chat_generate(self, input_string, temperature=0.0):
        """Generate a response using the chat format."""
        if isinstance(input_string, str):
            messages = [{"role": "user", "content": input_string}]
        else:
            messages = input_string

        try:
            if self.verbose:
                print(f"DeepSeekModel: Sending chat request to OpenRouter API")
                print(f"Model: {self.model_name}")
                print(f"Input length: {len(str(input_string))}")

            response = openrouter_chat_completions_with_backoff(
                api_key=self.api_key,
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=self.max_new_tokens,
                top_p=0.2,
                stop=self.stop_words,
                timeout=self.timeout,
                verbose=self.verbose
            )
            
            if self.verbose:
                print(f"DeepSeekModel: Received response from OpenRouter API")
                print(f"Response length: {len(str(response))}")
            
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error in chat_generate: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return ""
    
    def generate(self, input_string, temperature=0.0):
        """Generate a response (uses chat format for DeepSeek models)."""
        return self.chat_generate(input_string, temperature)
    
    async def batch_chat_generate(self, messages_list, temperature=0.0):
        """Generate responses for a batch of messages using the chat format."""
        try:
            if self.verbose:
                print(f"DeepSeekModel: Sending batch chat request to OpenRouter API")
                print(f"Model: {self.model_name}")
                print(f"Batch size: {len(messages_list)}")
            
            responses = await dispatch_openrouter_chat_requests(
                messages_list=messages_list,
                model=self.model_name,
                api_key=self.api_key,
                temperature=temperature,
                max_tokens=self.max_new_tokens,
                top_p=1.0,
                stop_words=self.stop_words,
                timeout=self.timeout,
                verbose=self.verbose
            )
            
            if self.verbose:
                print(f"DeepSeekModel: Received batch responses from OpenRouter API")
                print(f"Number of responses: {len(responses)}")
            
            return [response["choices"][0]["message"]["content"] for response in responses]
        except Exception as e:
            print(f"Error in batch_chat_generate: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return [""] * len(messages_list)
    
    async def batch_generate(self, messages_list, temperature=0.0):
        """Generate responses for a batch of messages (uses chat format for DeepSeek models)."""
        # Convert string inputs to chat format
        formatted_messages = []
        for input_item in messages_list:
            if isinstance(input_item, str):
                formatted_messages.append([{"role": "user", "content": input_item}])
            else:
                formatted_messages.append(input_item)
        
        return await self.batch_chat_generate(formatted_messages, temperature)
    
    def generate_insertion(self, input_string, suffix, temperature=0.0):
        """Generate a completion that comes after input_string and before suffix."""
        # DeepSeek doesn't have a direct insertion API, so we'll use a prompt-based approach
        prompt = f"{input_string}\n\n[Your response should be followed by this text: {suffix}]"
        response = self.generate(prompt, temperature)
        
        # Try to extract the part before the suffix
        if suffix in response:
            return response.split(suffix)[0]
        return response