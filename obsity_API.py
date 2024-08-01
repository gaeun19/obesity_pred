def table_definition_prompt(description: str) -> str:
    """Given the description of the dataframe results, generate a prompt for ChatGPT to provide solutions or insights based on the request

    Args:
        description (str): A description of the results of the dataframe, detailing the types of results or specific details.

    Returns:
        str: The prompt to be shown, which includes the description and a request for ChatGPT's analysis or solution.
    """
    prompt = f"""Given the following description of the dataframe results,
                provide an analysis or answer based on the request:
                
                \n### Description of DataFrame Results:
                {description}
                
                \n### Request:
                Based on the above description, provide an analysis or solution to the following question or request:
                """
    return prompt

# Description of the obesity categories
description = """
**Insufficient Weight (I):** This category indicates a state of underweight. Specifically, it means that the Body Mass Index (BMI) is below the normal range.

**Normal Weight (N):** This category signifies normal weight. It indicates that the BMI falls within the normal range.

**Overweight Level I (O-I):** This category represents a mild level of overweight. It means that the BMI is above the normal range but at an early stage of overweight.

**Overweight Level II (O-II):** This category denotes moderate overweight. It means that the BMI significantly exceeds the normal range, representing a middle stage of overweight.

**Obesity Type I (Ob-I):** This category refers to Type I obesity. It means that the BMI is above the lower threshold of the obesity range.

**Obesity Type II (Ob-II):** This category represents Type II obesity. It means that the BMI falls within the middle range of obesity criteria.

**Obesity Type III (Ob-III):** This category indicates Type III obesity. It signifies a severe level of obesity where the BMI exceeds the upper limit of the obesity range.
"""

    
