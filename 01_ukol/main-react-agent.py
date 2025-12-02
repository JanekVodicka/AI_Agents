from openai import OpenAI
import json
from typing import List, Dict, Any

# Client
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="none"
)

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


class ReactAgent:
    def __init__(self, model: str = "mistral"):
        self.model = model
        self.max_iterations = 10

    def run(self, messages: List[Dict[str, Any]]) -> str:

        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")

            # Call the LLM
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                parallel_tool_calls=False,
            )

            response_message = response.choices[0].message
            print(f"LLM Response: {response_message}")

            # Check if there are tool calls
            if response_message.tool_calls:
                # Add the assistant's message with tool calls to history
                messages.append(
                    {
                        "role": "assistant",
                        "content": response_message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in response_message.tool_calls
                        ],
                    }
                )

                # Process ALL tool calls (not just the first one)
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    tool_id = tool_call.id

                    print(f"Executing tool: {function_name}({function_args})")

                    # Call the function
                    function_to_call = available_functions[function_name]
                    function_response = function_to_call(**function_args)

                    print(f"Tool result: {function_response}")

                    # Add tool response to messages
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "name": function_name,
                            "content": json.dumps(function_response),
                        }
                    )

                # Continue the loop to get the next response
                continue

            else:
                # No tool calls - we have our final answer
                final_content = response_message.content

                # Add the final assistant message to history
                messages.append({"role": "assistant", "content": final_content})

                print(f"\nFinal answer: {final_content}")
                return final_content

        # If we hit max iterations, return an error
        return "Error: Maximum iterations reached without getting a final answer."


def main():
    # Create a ReAct agent
    agent = ReactAgent()

    # Example 1: Simple query (single tool call)
    print("=== Example 1: Single Tool Call ===")
    messages = [
                {"role": "developer", "content": "You are an AI assistant."},
                {"role": "user", "content": "Who is the current president of the United States?"}
            ]

    result1 = agent.run(messages.copy())
    print(f"\nResult: {result1}")


if __name__ == "__main__":
    main()
