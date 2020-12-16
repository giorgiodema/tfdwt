import tensorflow as tf


dict = {
    "uno":1.0,
    "due":2.0
}

@tf.function
def printnumber(inp):
    v = tf.Variable(0.0)
    #v.assign(dict[inp])
    print("tracing")
    return v

for i in range(10):
    if i%2==0:
        print(printnumber("uno"))
    else:
        print(printnumber("due"))