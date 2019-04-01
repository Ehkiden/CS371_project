from scapy.all import *
#from get_df import *
import pandas as pd
import numpy as np
import sys
import socket 
import os
    
def query(x):
  if IP in x:
    src_ip = x[IP].src
    dest_ip = x[IP].dst
    if x.haslayer(DNS) and x.getlayer(DNS).qr == 0:
      print(src_ip+" --> "+dest_ip+": "+x.getlayer(DNS).qd.qname)

def fields_extraction(x):
  query(x)
  print( x.sprintf("{IP:%IP.src%,%IP.dst%,}"
      "{TCP:%TCP.sport%,%TCP.dport%,%TCP.payload%}"
      "{UDP:%UDP.sport%,%UDP.dport%}"))
  print( x.summary())
  x.show()

pkts = sniff(prn = lambda x: fields_extraction(x), count = 50)

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
feature_3   -   chksum    -- use chksum to compare the send and recieve amount from src to dest. If src is recieving alot more than sending, then its downloading a file
feature_4   -   
feature_5   -   
label       -   1. Web browsing                     -   ports tcp/80,443
                2. Video streaming (e.g., YouTube)  -   uses TCP
                3. Video conference (e.g., Skype)   -   should be using UDP
                4. File download                    -   ports tcp/20, tcp/21? also tcp/443


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