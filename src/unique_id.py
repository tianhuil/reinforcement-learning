from random_word import RandomWords

r = RandomWords()
id_str = "{}-{}".format(r.get_random_word(), r.get_random_word())
print(id_str)
