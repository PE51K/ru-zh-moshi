import asyncio
import json
import os
import logging
import random
from datetime import datetime
from pathlib import Path
import openai
from tqdm.asyncio import tqdm
import signal

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

client = openai.OpenAI(
    api_key="anything", 
    base_url="http://127.0.0.1:8000/v1/", 
    default_headers={"Authorization": "Bearer anything"}
)

# Системная подсказка для изучения китайского языка
SYSTEM_PROMPT = """
Ты — преподаватель разговорного китайского языка.  

Твои задачи:  
- Помогать студентам изучать китайский язык.  
- Объяснять концепции языка коротко и ясно.  
- Предоставлять примеры и контекст, которые легко использовать в повседневной жизни.  
- Делать акцент на практическом применении языка.  
- Давать культурные пояснения только при необходимости.  
- Поддерживать активное общение с учеником.  

Формат ответа:  
- Отвечай кратко, как опытный преподаватель, объясняющий материал ученику.  
- Используй реальные примеры на китайском языке.  
- Ответы предоставляй только на русском или китайском языках.  
- Для китайского языка используй только иероглифы, никогда не пиши транскрипцию или пиньинь.  
- Отвечай строго в рамках своей роли.  
"""

# Подсказка для генерирования вопросов от пользователя
QUESTIONS_PROMPT = """
Ты — ученик, изучающий разговорный китайский язык. Твой родной язык — русский, а уровень владения китайским языком — начальный.

Твои задачи:
1. Формулировать разнообразные вопросы для преподавателя на русском языке, используя китайские иероглифы там, где это уместно.  
2. Интересоваться широким кругом тем, например:  
   - Грамматикой и правильным построением предложений.  
   - Использованием слов и выражений в разных ситуациях.  
   - Повседневной лексикой (еда, покупки, транспорт, семья, работа и т.д.).  
   - Разговорными фразами и устойчивыми выражениями.  
   - Разницей между формальным и неформальным стилем общения.  
   - Социальными и культурными нормами общения в Китае.  
   - Способами улучшения навыков понимания речи и говорения.  
3. Уточнять детали, приводить свои примеры или делиться опытом, чтобы углубить понимание языка.  

Правила и формат:
- Формулируй вопросы в стиле ученика, который активно изучает язык. Импровизируй с форматами вопросов, а не используй банальное "как правильно?". НЕ ИСПОЛЬЗУЙ "как правильно" в своей речи.
- Используй китайские иероглифы для примеров или уточнений.  
- Не используй транскрипцию или пиньинь.  
- Пиши только текст вопроса, без дополнительных комментариев.  
"""

def create_dataset_dir():
    """Create dataset directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_dir = Path("dataset") / timestamp
    dataset_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created dataset directory: {dataset_dir}")
    return dataset_dir

async def generate_conversation():
    """Generate one conversation with multiple turns"""
    try:
        turns = random.randint(2, 3)  # Random number of turns between 2 and 3
        conversation = []

        # Generate initial question
        logger.debug("Generating initial question")
        response = await asyncio.wait_for(
            asyncio.to_thread(
                client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": QUESTIONS_PROMPT},
                    {"role": "user", "content": "Сформулируй вопрос"},
                ]
            ),
            timeout=60  # 60 seconds timeout
        )
        user_question = response.choices[0].message.content.strip()
        logger.debug(f"Generated initial question: {user_question}")
        
        # Generate assistant response
        logger.debug("Generating assistant response")
        response = await asyncio.wait_for(
            asyncio.to_thread(
                client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_question}
                ]
            ),
            timeout=60  # 60 seconds timeout
        )
        assistant_response = response.choices[0].message.content.strip()
        logger.debug(f"Generated assistant response: {assistant_response}")

        # Add initial turn to conversation
        conversation.extend([
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": assistant_response}
        ])

        # Generate follow-up turns
        for turn in range(turns - 1):
            logger.debug(f"Generating follow-up question {turn + 2}")
            
            # Build messages array with conversation history for follow-up question
            messages = [
                {"role": "system", "content": QUESTIONS_PROMPT},
                *conversation,  # Include full conversation history
                {"role": "user", "content": "Задай следующий вопрос"},
            ]
            
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    client.chat.completions.create,
                    model="gpt-4o-mini",
                    messages=messages
                ),
                timeout=60  # 60 seconds timeout
            )
            user_question = response.choices[0].message.content.strip()
            logger.debug(f"Generated follow-up question: {user_question}")

            # Build messages array with conversation history for assistant response
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                *conversation,  # Include full conversation history
                {"role": "user", "content": user_question}
            ]
            
            # Generate assistant response for follow-up
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    client.chat.completions.create,
                    model="gpt-4o-mini",
                    messages=messages
                ),
                timeout=60  # 60 seconds timeout
            )
            assistant_response = response.choices[0].message.content.strip()
            logger.debug(f"Generated follow-up response: {assistant_response}")

            # Add turn to conversation
            conversation.extend([
                {"role": "user", "content": user_question},
                {"role": "assistant", "content": assistant_response}
            ])

        return {"messages": conversation}
    except asyncio.TimeoutError:
        logger.error("Timeout while generating conversation")
        return None
    except Exception as e:
        logger.error(f"Error generating conversation: {e}")
        return None

async def generate_dataset(num_items=1, num_workers=1):
    """Generate dataset with multiple workers"""
    dataset_dir = create_dataset_dir()
    
    # Save system prompt
    with open(dataset_dir / "system_prompt.json", "w", encoding="utf-8") as f:
        json.dump({"role": "system", "content": SYSTEM_PROMPT}, f, ensure_ascii=False, indent=2)
    logger.info("Saved system prompt")
    
    # Create task queue
    queue = asyncio.Queue()
    for i in range(num_items):
        queue.put_nowait(i)
    
    # Create progress bar
    pbar = tqdm(total=num_items, desc="Generating conversations")
    
    # Shutdown flag for graceful termination
    shutdown = asyncio.Event()
    
    async def worker():
        while not shutdown.is_set():
            try:
                item_num = queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            try:
                conv = await generate_conversation()
                if conv and not shutdown.is_set():
                    file_path = dataset_dir / f"conversation_{item_num:04d}.json"
                    logger.debug(f"Attempting to save conversation with {len(conv['messages'])} messages")
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(
                            conv["messages"],
                            f, 
                            ensure_ascii=False, 
                            indent=2
                        )
                    logger.info(f"Successfully saved {file_path}")
                    pbar.update(1)
            except Exception as e:
                logger.error(f"Error processing conversation {item_num}: {e}")
            finally:
                queue.task_done()

    # Handle shutdown signals
    def signal_handler():
        logger.info("Shutdown signal received")
        shutdown.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        # Create and run workers
        workers = [asyncio.create_task(worker()) for _ in range(num_workers)]
        
        # Wait for queue to be processed or shutdown signal
        try:
            await asyncio.wait_for(queue.join(), timeout=num_items * 120)  # 120 seconds per item timeout
        except asyncio.TimeoutError:
            logger.warning("Operation timed out")
            shutdown.set()
        
        # Cancel pending tasks
        for w in workers:
            w.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*workers, return_exceptions=True)
        
    finally:
        # Cleanup
        pbar.close()
        logger.info(f"Generation completed. Results saved in {dataset_dir}")
        
        # Remove signal handlers
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)
    
    return dataset_dir

if __name__ == "__main__":
    num_conversations = 1000  # For testing, generate just one conversation first
    num_workers = 2        # Single worker for testing
    
    try:
        asyncio.run(generate_dataset(num_conversations, num_workers))
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Process failed: {e}")
