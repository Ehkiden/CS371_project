from scapy.all import sniff
#from get_df import *
import pandas as pd
import numpy as np
import sys
import socket 
import os
    
def fields_extraction(x):
     print( x.sprintf("{IP:%IP.src%,%IP.dst%,}"
         "{TCP:%TCP.sport%,%TCP.dport%,}"
         "{UDP:%UDP.sport%,%UDP.dport%}"))
     print( x.summary())
     x.show()

pkts = sniff(prn = lambda x: fields_extraction(x), count = 100)

#"show" function 


'''
what to extract:
src_ip
src_port
dest_ip
dest_port
app(from protocol, len, and port(from host)) 

columns:
flow_id     -   auto incremented
IPsrc       -   extracted
IPdst       -   extracted
proto       -   extracted
feature_1   -   src_port
feature_2   -   dest_port
feature_3   -   
feature_4   -   
feature_5   -   
label       -   


'''

'''
example packet from twicth streams

52.223.224.71,192.168.1.155,https,11203,
Ether / IP / TCP 52.223.224.71:https > 192.168.1.155:11203 A / Raw
###[ Ethernet ]###
  dst       = 08:62:66:2c:4a:a1
  src       = c0:56:27:6d:97:87
  type      = 0x800
###[ IP ]###
     version   = 4
     ihl       = 5
     tos       = 0x0
     len       = 1500
     id        = 11905
     flags     = DF
     frag      = 0
     ttl       = 53
     proto     = tcp
     chksum    = 0x3a31
     src       = 52.223.224.71
     dst       = 192.168.1.155
     \options   
###[ TCP ]###
        sport     = https
        dport     = 11203
        seq       = 307931811
        ack       = 4185644134
        dataofs   = 5
        reserved  = 0
        flags     = A
        window    = 363
        chksum    = 0x4b10
        urgptr    = 0
        options   = []
'''