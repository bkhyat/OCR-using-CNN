def get_class_label_from_index(ind):
    classes = list()
    indices = list()
    category_file = open('category.txt', 'r')
    x = category_file.readlines()
    category_file.close()
    x = [item.strip() for item in x]
    for each in x:
        index, Class = each.split(";")
        classes.append(Class)
        indices.append(int(index))
    return classes[indices.index(ind)]

