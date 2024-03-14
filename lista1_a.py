'''
link: https://drive.google.com/drive/folders/1gMFshl41F5RZTk6-IYgrFuqLTq9kUFMK

(a) Implemente um algoritmo genético simples (como definido na seção 4) em Python
ou em sua linguagem preferida para maximizar a função dada em (1), ou seja:

f(y) = y + |sin(32y)|, 0 ≤ y < π

Use as funções get_bits e get_float em Python apresentadas abaixo para converter
números reais em cadeias de bits e vice-versa (ou faça uma implementação própria
sem usar strings).
'''

L = 4 * 8 # size of chromossome in bits

import struct

def floatToBits ( f ) :
    s = struct . pack ('>f', f )
    return struct . unpack ('>L', s ) [0]

def bitsToFloat ( b ) :
    s = struct . pack ('>L', b )
    return struct . unpack ('>f', s ) [0]

# Exemplo : 1.23 -> '00010111100 '
def get_bits ( x ) :
    x = floatToBits ( x )
    N = 4 * 8
    bits = ''
    for bit in range ( N ) :
        b = x & (2** bit )
        bits += '1' if b > 0 else '0'
    return bits

# Exemplo : '00010111100 ' -> 1.23
def get_float ( bits ) :
    x = 0
    assert ( len( bits ) == L )
    for i , bit in enumerate ( bits ) :
        bit = int ( bit ) # 0 or 1
        x += bit * (2** i )
    return bitsToFloat ( x)