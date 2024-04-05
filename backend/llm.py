from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_response(product_code: str, warehouse: str, product_category: str,
                      date: str, forecast: int, max_length: int=150, repetition_penalty: float=1.2,
                      num_return_sequences: int=1, temperature: float=0.6, top_k: int=50):
  
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    if forecast > 500:
        action = 'Consider increasing the stock or production'
    elif forecast <= 500 and forecast > 300:
        action = 'Maintain the production and stock levels'
    else:
        action = 'Consider reducing the stock or production'

    prompt = f"The forecasted demand for product {product_code} from warehouse {warehouse} under category {product_category} \
    at date {date} is approximately {forecast}. Based on this, the recommended action is: {action}."
  
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
  
    output = model.generate(input_ids, 
        do_sample=True, 
        max_length=max_length, 
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        top_k=top_k, 
        num_return_sequences=num_return_sequences
    )
    return [tokenizer.decode(output[i], skip_special_tokens=True) for i in range(num_return_sequences)]