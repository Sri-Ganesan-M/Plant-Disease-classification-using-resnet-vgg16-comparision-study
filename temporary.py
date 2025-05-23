import google.generativeai as genai

def get_precaution(name):    
    genai.configure(api_key='')
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content("give me in 10 points What is the precaution for the plant disease " + name)
    data = response.text

    # Remove the unwanted '*' and '>' characters before ''
    data = data.replace('> *', '').replace('*', '')

    return data

# Example usage:
precautions = get_precaution("your disease name here")
print(precautions)
