import os
from dotenv import load_dotenv
import openai
from openai import OpenAI
import anthropic
import google.generativeai as genai
import json
import re
import random
from tqdm import tqdm
import time
from datasets import load_dataset
import argparse
import requests
from ai21 import AI21Client
from ai21.models.chat import ChatMessage, ResponseFormat, DocumentSchema, FunctionToolDefinition
from ai21.models.chat import ToolDefinition, ToolParameters
import os

load_dotenv()

ENV_API_KEY = os.getenv("OPENAI_API_KEY")
JAMBA_API_KEY = os.getenv("JAMBA_API_KEY")

if ENV_API_KEY is None:
    raise ValueError("OPENAI_API_KEY is not set in environment variables")

API_KEY = JAMBA_API_KEY

def get_client():
    if args.model_name in ["gpt-4", "gpt-4o", "o1-preview"]:
        openai.api_key = API_KEY
        client = openai
    elif args.model_name in ["deepseek-chat", "deepseek-coder"]:
        client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com/")
    elif args.model_name in ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest",
                             "gemini-1.5-flash-8b", "gemini-002-pro", "gemini-002-flash"]:
        genai.configure(api_key=API_KEY)
        generation_config = {
            "temperature": 0.0,
            "top_p": 1,
            "max_output_tokens": 4000,
            "response_mime_type": "text/plain",
        }
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]
        client = genai.GenerativeModel(
            model_name=args.model_name,
            safety_settings=safety_settings,
            generation_config=generation_config,
        )
    elif args.model_name in ["claude-3-opus-20240229", "claude-3-sonnet-20240229"]:
        client = anthropic.Anthropic(
            api_key=API_KEY,
        )
    elif args.model_name in ["jamba-1.5-large", "jamba-1.5-mini"]:
        client = AI21Client(api_key=API_KEY)
    elif args.model_name in ["iask"]:
        client = {"Authorization": f"Bearer {API_KEY}"}
    else:
        client = None
        print("For other model API calls, please implement the client definition method yourself.")
    return client


def call_api(client, instruction, inputs):
    start = time.time()
    if args.model_name in ["gpt-4", "gpt-4o", "deepseek-chat", "deepseek-coder"]:
        message_text = [{"role": "user", "content": instruction + inputs}]
        completion = client.chat.completions.create(
          model=args.model_name,
          messages=message_text,
          temperature=0,
          max_tokens=4000,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0,
        )
        result = completion.choices[0].message.content
    elif args.model_name in ["o1-preview"]:
        message_text = [{"role": "user", "content": instruction + inputs}]
        completion = client.chat.completions.create(
          model=args.model_name,
          messages=message_text,
        )
        result = completion.choices[0].message.content
    elif args.model_name in ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest", "gemini-1.5-flash-8b"]:
        chat_session = client.start_chat(
            history=[]
        )
        result = chat_session.send_message(instruction + inputs).text
    elif args.model_name in ["claude-3-opus-20240229", "claude-3-sonnet-20240229"]:
        message = client.messages.create(
            model=args.model_name,
            max_tokens=4000,
            system="",
            messages=[
                {"role": "user", "content": instruction + inputs}
            ],
            temperature=0.0,
            top_p=1,
        )
        result = message.content[0].text
    elif args.model_name in ["jamba-1.5-large", "jamba-1.5-mini"]:
        message_text = [ChatMessage(content=instruction + inputs, role="user")]
        completion = client.chat.completions.create(
            model=args.model_name,
            messages=message_text,
            documents=[],
            tools=[],
            n=1,
            max_tokens=2048,
            temperature=0,
            top_p=1,
            stop=[],
            response_format=ResponseFormat(type="text"),
        )
        result = completion.choices[0].message.content
    elif args.model_name in ["iask"]:
        payload = {
            "prompt": instruction + inputs,
            "mode": "truth",
            "detail_level": "detailed",
            "stream": False
        }
        response = requests.post("https://api.iask.ai/v1/query", headers=client, json=payload, timeout=300)
        if response.status_code != 200:
            print("API call failed with status code", response.status_code, response.json())
            return response.json()["response"]["message"]
        else:
            result = response.json()["response"]["message"]
        return result
    else:
        print("For other model API calls, please implement the request method yourself.")
        result = None
    print("cost time", time.time() - start)
    return result


def load_mmlu_pro():
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    test_df, val_df = dataset["test"], dataset["validation"]
    
    test_df = preprocess(test_df, args.questions_per_topic)
    val_df = preprocess(val_df)
    return test_df, val_df


def preprocess(test_df, questions_per_topic=None):
    res_df = []
    for each in test_df:
        options = []
        for opt in each["options"]:
            if opt == "N/A":
                continue
            options.append(opt)
        each["options"] = options
        res_df.append(each)
    res = {}
    for each in res_df:
        if each["category"] not in res:
            res[each["category"]] = []
        res[each["category"]].append(each)
    
    # Limit questions per topic if specified
    if questions_per_topic:
        for category in res:
            if len(res[category]) > questions_per_topic:
                res[category] = res[category][:questions_per_topic]
    
    return res


def format_example(question, options, cot_content=""):
    if cot_content == "":
        cot_content = "Let's think step by step."
    if cot_content.startswith("A: "):
        cot_content = cot_content[3:]
    example = "Question: {}\nOptions: ".format(question)
    choice_map = "ABCDEFGHIJ"
    for i, opt in enumerate(options):
        example += "{}. {}\n".format(choice_map[i], opt)
    if cot_content == "":
        example += "Answer: "
    else:
        example += "Answer: " + cot_content + "\n\n"
    return example


def extract_answer(text):
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        print("1st answer extract failed\n" + text)
        return extract_again(text)


def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)


def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None


def single_request(client, single_question, cot_examples_dict, exist_result):
    """
    Makes a single API request to get model prediction for a question.
    
    Args:
        client: API client instance
        single_question (dict): Question data containing id, category, question text and options
        cot_examples_dict (dict): Chain-of-thought examples organized by category
        exist_result (list): Previously cached results to check for existing predictions
    
    Returns:
        tuple: (prediction, model output text, whether result existed in cache)
            prediction: Single letter A-J indicating predicted answer
            model_output: Full response text from model
            exist: Boolean indicating if result was found in cache
    """
    # Check if question already exists in cached results
    exist = True
    q_id = single_question["question_id"]
    for each in exist_result:
        if q_id == each["question_id"] and single_question["question"] == each["question"]:
            pred = extract_answer(each["model_outputs"])
            return pred, each["model_outputs"], exist
    
    # Question not found in cache, make new API request
    exist = False
    category = single_question["category"]
    cot_examples = cot_examples_dict[category]
    question = single_question["question"]
    options = single_question["options"]
    
    # Construct prompt with examples and instructions
    prompt = "The following are multiple choice questions (with answers) about {}. Think step by" \
             " step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n" \
        .format(category)
    for each in cot_examples:
        prompt += format_example(each["question"], each["options"], each["cot_content"])
    input_text = format_example(question, options)
    
    # Make API call and process response
    try:
        response = call_api(client, prompt, input_text)
        response = response.replace('**', '')
    except Exception as e:
        print("error", e)
        return None, None, exist
        
    pred = extract_answer(response)
    return pred, response, exist


def update_result(output_res_path):
    """
    Updates evaluation results and category statistics from saved results file.
    
    Args:
        output_res_path (str): Path to JSON file containing evaluation results
        
    Returns:
        tuple: (results list, category statistics dict)
            results: List of evaluation results for each question
            category_record: Dict mapping categories to correct/wrong counts
    """
    category_record = {}
    res = []
    success = False
    
    # Keep trying to load and process results until successful
    while not success:
        try:
            if os.path.exists(output_res_path):
                with open(output_res_path, "r") as fi:
                    res = json.load(fi)
                    # Calculate statistics for each category
                    for each in res:
                        category = each["category"]
                        if category not in category_record:
                            category_record[category] = {"corr": 0.0, "wrong": 0.0}
                            
                        # Handle missing predictions with random guess
                        if not each["pred"]:
                            random.seed(12345)
                            x = random.randint(0, len(each["options"]) - 1)
                            if x == each["answer_index"]:
                                category_record[category]["corr"] += 1
                            else:
                                category_record[category]["wrong"] += 1
                        # Update correct/wrong counts based on prediction
                        elif each["pred"] == each["answer"]:
                            category_record[category]["corr"] += 1
                        else:
                            category_record[category]["wrong"] += 1
            success = True
        except Exception as e:
            print("Error", e, "sleep 2 seconds")
            time.sleep(2)
    return res, category_record


def merge_result(res, curr):
    """
    Merges a new result into existing results list, updating if question exists.
    
    Args:
        res (list): Existing results list
        curr (dict): New result to merge in
        
    Returns:
        list: Updated results list with new result merged in
    """
    merged = False
    # Try to find and update existing question
    for i, single in enumerate(res):
        if single["question_id"] == curr["question_id"] and single["question"] == curr["question"]:
            res[i] = curr
            merged = True
    # Append if question not found
    if not merged:
        res.append(curr)
    return res


def evaluate(subjects):
    """
    Evaluate model performance on MMLU benchmark across specified subjects.
    
    Args:
        subjects (list): List of subject names to evaluate. If empty, evaluates all subjects.
    
    This function:
    1. Initializes API client and loads test/dev datasets
    2. For each subject:
        - Processes each test question
        - Makes API requests to get model predictions
        - Updates and saves results and summary statistics
    
    Results are saved in two files per subject:
    - {model_name}_{subject}_result_{timestamp}.json: Detailed results for each question
    - {model_name}_{subject}_summary_{timestamp}.json: Accuracy statistics summary
    """
    # Initialize API client and load datasets
    client = get_client()
    test_df, dev_df = load_mmlu_pro()
    
    # If no subjects specified, evaluate all subjects
    if not subjects:
        subjects = list(test_df.keys())
    print("assigned subjects", subjects)

    # Get current timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Add overall statistics tracking
    overall_stats = {"corr": 0.0, "wrong": 0.0}
    overall_output_path = os.path.join(args.output_dir, f"{args.model_name}_overall_summary_{timestamp}.json")

    # Process each subject
    for subject in subjects:
        test_data = test_df[subject]
        output_res_path = os.path.join(args.output_dir, f"{args.model_name}_{subject}_result_{timestamp}.json")
        output_summary_path = os.path.join(args.output_dir, f"{args.model_name}_{subject}_summary_{timestamp}.json")
        res, category_record = update_result(output_res_path)

        # Process each question in the test set
        for each in tqdm(test_data):
            label = each["answer"]
            category = subject
            
            # Get model prediction
            pred, response, exist = single_request(client, each, dev_df, res)
            
            if response is not None:
                # Update results and records
                res, category_record = update_result(output_res_path)
                if category not in category_record:
                    category_record[category] = {"corr": 0.0, "wrong": 0.0}
                
                # Store prediction and model output
                each["pred"] = pred
                each["model_outputs"] = response
                merge_result(res, each)
                
                # Update accuracy counts
                if pred is not None:
                    if pred == label:
                        category_record[category]["corr"] += 1
                        overall_stats["corr"] += 1
                    else:
                        category_record[category]["wrong"] += 1
                        overall_stats["wrong"] += 1
                else:
                    category_record[category]["wrong"] += 1
                    overall_stats["wrong"] += 1
                    
                # Save intermediate results
                save_res(res, output_res_path)
                save_summary(category_record, output_summary_path)
                res, category_record = update_result(output_res_path)
        
        
        # Save subject-specific results
        save_res(res, output_res_path)
        save_summary(category_record, output_summary_path)
    
    # Save overall summary using save_summary function
    print("overall_stats", overall_stats)
    save_overall_summary(overall_stats, overall_output_path)

def save_res(res, output_res_path):
    """
    Save evaluation results to a JSON file after removing duplicate question IDs.
    
    Args:
        res (list): List of dictionaries containing evaluation results for each question
        output_res_path (str): Path to save the output JSON file
    
    The function:
    1. Removes duplicate entries based on question_id
    2. Keeps only the first occurrence of each question
    3. Saves the deduplicated results to the specified JSON file
    """
    # Remove duplicates while preserving order
    temp = []
    exist_q_id = []
    for each in res:
        if each["question_id"] not in exist_q_id:
            exist_q_id.append(each["question_id"])
            temp.append(each)
        else:
            continue
    res = temp
    
    # Save to JSON file
    with open(output_res_path, "w") as fo:
        fo.write(json.dumps(res))
        
def save_overall_summary(overall_stats, overall_output_path):
    total = overall_stats["corr"] + overall_stats["wrong"]
    if total > 0:
        overall_stats["acc"] = overall_stats["corr"] / total
    else:
        overall_stats["acc"] = 0.0
    
    with open(overall_output_path, "w") as fo:
        fo.write(json.dumps(overall_stats))

def save_summary(category_record, output_summary_path):
    """
    Calculate and save evaluation summary statistics to a JSON file.
    
    Args:
        category_record (dict): Dictionary containing correct/wrong counts for each category
        output_summary_path (str): Path to save the summary JSON file
    
    The function:
    1. Calculates accuracy for each category
    2. Computes overall totals and accuracy across all categories
    3. Saves the summary statistics to the specified JSON file
    
    The output JSON contains per-category and total:
    - Number of correct answers (corr)
    - Number of wrong answers (wrong) 
    - Accuracy (acc)
    """
    total_corr = 0.0
    total_wrong = 0.0
    
    # Calculate per-category accuracy and accumulate totals
    for k, v in category_record.items():
        if k == "total":
            continue
        cat_acc = v["corr"] / (v["corr"] + v["wrong"])
        category_record[k]["acc"] = cat_acc
        total_corr += v["corr"]
        total_wrong += v["wrong"]
    
    # Calculate and add overall accuracy
    acc = total_corr / (total_corr + total_wrong)
    category_record["total"] = {"corr": total_corr, "wrong": total_wrong, "acc": acc}
    
    # Save to JSON file
    with open(output_summary_path, "w") as fo:
        fo.write(json.dumps(category_record))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "-o", type=str, default="eval_results/")
    parser.add_argument("--questions_per_topic", "-q", type=int, default=0,
                        help="Number of questions to use per topic (0 for all questions)")
    parser.add_argument("--model_name", "-m", type=str, default="gpt-4",
                        choices=["gpt-4", "gpt-4o", "o1-preview",
                                 "deepseek-chat", "deepseek-coder",
                                 "gemini-1.5-flash-latest",
                                 "gemini-1.5-pro-latest",
                                 "claude-3-opus-20240229",
                                 "gemini-1.5-flash-8b",
                                 "claude-3-sonnet-20240229",
                                 "gemini-002-pro",
                                 "gemini-002-flash",
                                 "jamba-1.5-large",
                                 "jamba-1.5-mini"])
    parser.add_argument("--assigned_subjects", "-a", type=str, default="all")
    assigned_subjects = []
    args = parser.parse_args()

    if args.assigned_subjects == "all":
        assigned_subjects = []
    else:
        assigned_subjects = args.assigned_subjects.split(",")
    os.makedirs(args.output_dir, exist_ok=True)
    evaluate(assigned_subjects)
