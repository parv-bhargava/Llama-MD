import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
os.chdir('../')

# Use the below 3 lines for base model
# model_name = "meta-llama/Llama-3.2-1B"
# model = AutoModelForCausalLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# Use the below 2 lines if you trained your model and want to use it
# model = AutoModelForCausalLM.from_pretrained("./fine_tuned_modelpf_lora_epoch_5/")
# tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_modelpf_lora_epoch_5/")

# Use the below lines for loading out model for inferencing
model = AutoModelForCausalLM.from_pretrained("./models/")
tokenizer = AutoTokenizer.from_pretrained("./models/")
model.to(device)

def doctor_response(question, max_length=300, temperature=0.6):
    """
    Generates a doctor's response based on the patient's question.

    Parameters:
        question (str): The patient's question.
        max_length (int): The maximum token length of the response.
        temperature (float): Sampling temperature for diversity.
        top_p (float): Nucleus sampling parameter.

    Returns:
        str: The doctor's response.
    """
    prompt = (
        "<System Prompt>\n"
        "You are an expert medical doctor of the patient.\n"
        "Read the patient's query and provide a clear, concise, and medically sound response.\n\n"
        "Your answer should include:\n"
        "- A diagnosis\n"
        "- A recommended treatment plan or next steps\n\n"
        "Do not repeat the patient's question. Avoid unnecessary disclaimers.\n"
        "Keep your answer focused, authoritative, and helpful.\n"
        "</System Prompt>\n\n"
        "Query: {question}\n\n"
        "Your Response:".format(question=question)
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    # Remove the input text from the generated output
    generated_tokens = outputs[0]
    input_length = inputs["input_ids"].shape[1]
    output_final = generated_tokens[input_length:]
    response = tokenizer.decode(output_final, skip_special_tokens=True)
    return response # Extract the response after "Doctor:"


# Example usage
if __name__ == "__main__":
    print("Welcome to you Virtual gynecologist- LLAMA, MD")
    # user_input = input("You (Patient): ")
    # user_input = "I randomly get nauseatic and I have been constantly vomiting for the last few days. I also have a mild fever. What do you think is wrong with me?"
    # user_input = "Hi doctor,I am just wondering what is abutting and abutment of the nerve root means in a back issue. Please explain. What treatment is required for annular bulging and tear?"
    user_input = "Hi doctor, what food should I eat in my third trimester?"
    # user_input = "I have a sharp pain in my chest that radiates to my left arm. I feel dizzy and short of breath. What should I do?"
    answer = doctor_response(user_input)
    print(answer)
