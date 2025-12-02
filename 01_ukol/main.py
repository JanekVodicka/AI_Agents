from openai import OpenAI
import json

# --- Function Implamentation ---
def get_current_president(country: str):
    presidents = {
        "United States": "Donald Trump",
        "France": "Emmanuel Macron",
        "Germany": "Frank-Walter Steinmeier"  # jen příklad
    }
    current_president = presidents.get(country, "Unknown")
    return {"country": country, "current_president": current_president}

def get_president_party(country: str):
    parties = {
        "United States": "Republic Party",
        "France": "La République En Marche",
        "Germany": "Social Democratic Party"
    }
    president_party = parties.get(country, "Unknown")
    return {"country": country, "president_party": president_party}


# Tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_president",
            "description": "Use this function to get the current president of the United States.",
            "parameters": {
                "type": "object",
                "properties": {
                    "country": {
                        "type": "string",
                        "description": "The country to get the president for, e.g. 'United States'",
                    }
                },
                "required": ["country"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_president_party",
            "description": "Use this function to get the political party of the current president.",
            "parameters": {
                "type": "object",
                "properties": {
                    "country": {
                        "type": "string",
                        "description": "The country to get the president's party for, e.g. 'United States'",
                    }
                },
                "required": ["country"],
            },
        },
    },
]

# Function dictionary
available_functions = {
    "get_current_president": get_current_president,
    "get_president_party": get_president_party,
}

# Nasazení na Ollamu
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="none"
)

messages = [
            {"role": "developer", "content": "You are an AI assistant."},
            {"role": "user", "content": "Who is the current president of the United States?"}
        ]

model = "mistral"

for i in range(1):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    # print(response.to_json())

    response_message = response.choices[0].message
    if response_message.tool_calls:
        print("Response obsahuje tool call!!")
        print(50*"-")

        # Find the tool call content
        tool_call = response_message.tool_calls[0]
        print("Celý toolcall: ", tool_call)

        # Extract tool name, id and arguments
        function_name = tool_call.function.name
        print("Function name: ", function_name)
        tool_id = tool_call.id
        print("Tool id: ", tool_id)
        function_args = json.loads(tool_call.function.arguments)                        # Ze stringu udělá dictionary argumentů
        print("Argumenty funkce jsou: ", function_args)

        # Call the function
        function_to_call = available_functions[function_name]
        print("Vybraná funkce pro návratovou message:", function_to_call.__name__)

        function_response = function_to_call(**function_args)                           # ** = rozbalení argumentů
        print("Výsledek vybrané funkce: ", function_response)

        messages.append({
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tool_id,  
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": json.dumps(function_args),
                    }
                }
            ]
        })
        messages.append({
            "role": "tool",
            "tool_call_id": tool_id,  
            "name": function_name,
            "content": json.dumps(function_response),
        })

        # Second call to get final response based on function output
        second_response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,  
            tool_choice="auto"  
        )
        final_answer = second_response.choices[0].message
        print(50*"-")
        if final_answer.tool_calls:
            print("Final response obsahuje tool call!!")
        else:
            print("Final response NEobsahuje tool call!!")

        print(50*"-")
        print("Final response:", final_answer.content)
        # return final_answer
    else:
        print("Není žádný toolcall")
        print(response_message)