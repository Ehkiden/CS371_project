from scapy.all import sniff
import pandas as pd
import numpy as np
import sys
import socket 
import os
    
def fields_extraction(x):
  #current format for tcp:
  #src_ip, dest_ip, len, protocol, src_port, dest_port, seq, ack
  #current format for udp:
  #src_ip, dest_ip, len, protocol, src_port, dest_port, len
  print( x.sprintf("{IP:%IP.src%,%IP.dst%,%IP.len%,}"
      "{TCP:tcp,%TCP.sport%,%TCP.dport%,%TCP.seq%,%TCP.ack%}"
      "{UDP:udp,%UDP.sport%,%UDP.dport%,%UDP.len%}"))
  #print( x.summary())
  #x.show()
  #

pkts = sniff(filter="port 443 or port 80" ,prn = lambda x: fields_extraction(x), count = 50)

#"show" function 


'''
Soooooooooo
What they want is a set of packets to represent a flow
meaning that each flow will contain packets where the src_ip, dest_ip, protocol, src_port, dest_port are crossed
ref with each other or the same

This would look like:
duration(not needed), src_ip, dest_ip, protocol, src_port, dest_port, bytes_in, bytes_out, seq, ack, src_transfer_rate, dest_transfer_rate, 

src_ip    - This will be our ip (for simplicity sake...)
src_port  - The src ip's port
dest_ip   - The dest ip that is seen with the src ip
dest_port - The dest port
protocol  - TCP or UDP
bytes_in  - The total bytes the src_ip recieves from the dest_ip (aka the len )
bytes_out - The total bytes the src_ip sent to the dest_ip (aka the len )
**************** maybe skip these and just go by ip and port
seq       - The sequence number which identifies if the session is current along with the ack number  
ack       - The ack # used with the seq # to identify is the session is current
****************
src_transfer_rate   - Just the percentage of bytes_in/(bytes_in + bytes_out)
dest_transfer_rate  - Just the percentage of bytes_out/(bytes_in + bytes_out)

This should be good enough for what we need ^
----------------------------------------------
The labels will be applied after we have both packet flows into the tuple(which goes into the .csv)

As for label defs:

* Web Browsing should always use 80/tcp and/or 443/tcp and be a relatively small byte count
* Video Streaming will use tcp as well as udp depending on the type of stream. Youtube uses tcp for thier prerendered videos 
  while Live streaming sites will use UDP as they are no prerendered 
* Video Conference will use UDP
* File download via browser will use the same method as web browsing and stuff like scp will use 21/tcp and 22/tcp

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

