'''o200k_base: Used by newer models like gpt-4o and gpt-4o-mini.
   cl100k_base: Widely used by popular models such as gpt-3.5-turbo, gpt-4-turbo, gpt-4, and several embedding models.
   p50k_base: Employed by Codex models, as well as text-davinci-002 and text-davinci-003.
   r50k_base (also known as gpt2): Used by older GPT-3 models, including those named davinci.'''

import tiktoken

encodings_list = ['o200k_base', 'cl100k_base', 'p50k_base', 'r50k_base']
text = input("Enter a string you want to check: ")
for enc_name in encodings_list:
    encoding = tiktoken.get_encoding(enc_name)
    tokens_enc = encoding.encode(text)
    dec_text = encoding.decode(tokens_enc)
    tokens_dec = encoding.encode(dec_text)
    enc_token_count = len(tokens_enc)
    dec_token_count = len(tokens_dec)
    total_total_count = enc_token_count + dec_token_count
    print("Checking the Encoding: ",enc_name)
    print("User Text: ",text)
    print("Decoded Text: ", dec_text)
    print("Enc Token ID's: ",tokens_enc)
    print("Dec Token ID's:", tokens_dec)
    print("Encoding Token Count: ", enc_token_count)
    print("Decoding Token Count: ", dec_token_count)
    print("Total Tokens: ", total_total_count)
    print("-----------------------------------------------")