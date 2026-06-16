from lib.InvertedIndex import InvertedIndex

def build_command():
    invIdx = InvertedIndex()
    invIdx.build()
    invIdx.save()
