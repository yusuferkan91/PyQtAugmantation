text = "|__________________|"
list_text = list(text)
txt = "asdsdfdfsaddasdsad"
list_txt = list(txt)
l_text = len(text)
l_txt = len(txt)
index = int((l_text-l_txt)/2)

for each in range(len(txt)):
    list_text[index+each] = list_txt[each]
print(list_text)
str1 = ''.join(list_text)
print(text)
print(str1)
