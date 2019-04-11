from scapy.all import sniff
#import pandas as pd
import numpy as np
import sys
import socket 
import os
import time
    

# extracts the fields and applies them to the array
def fields_extraction(x, results, flowList):

  try:
    #assign the variables
    src_ip=(str(x.sprintf("{IP:%IP.src%}")))
    src_port=str(x.sprintf("{TCP:%TCP.sport%}""{UDP:%UDP.sport%}"))
    dest_ip=str(x.sprintf("{IP:%IP.dst%}"))
    dest_port=str(x.sprintf("{TCP:%TCP.dport%}""{UDP:%UDP.dport%}"))
    proto=str(x.sprintf("{TCP:tcp}""{UDP:udp}"))
    pkt_bytes=int(x.sprintf("{IP:%IP.len%}"))

    #check if the flowList list is empty or not
    if(len(flowList)!=0):
      #iterate through the tables
      flow_match=0  #initiate a var to determine if matching flow was found
      for flow in flowList:
        if( src_ip == flow[1] and dest_ip == flow[2] and src_port == flow[3] and dest_port == flow[4] and proto == flow[5] ):
          #means we sent the pkt
          #update the tuple bytes and pkts
          flow[6] = flow[6] + 1           #total pkts
          flow[7] = flow[7] + 1           #src pkts
          flow[9] = flow[9] + pkt_bytes   #total bytes
          flow[10] = flow[10] + pkt_bytes #src bytes
          flow[13] = int(time.time())-flow[12]  #increase the duration
          flow_match=1


        elif( src_ip == flow[2] and dest_ip == flow[1] and src_port == flow[4] and dest_port == flow[3] and proto == flow[5] ):
          #means the we recieved the pkt
          #update the tuple bytes and pkts
          flow[6] = flow[6] + 1           #total pkts
          flow[8] = flow[8] + 1           #dest pkts
          flow[9] = flow[9] + pkt_bytes   #total bytes
          flow[11] = flow[11] + pkt_bytes #dest bytes
          flow[13] = int(time.time())-flow[12]  #increase the duration
          flow_match=1

      if(flow_match==0): #add a new flow id to the list
        addFlow=[src_ip, dest_ip, src_port, dest_port, proto, 1, 1, 0, pkt_bytes, pkt_bytes, 0, int(time.time()), int(time.time()), 0]
        flowList.append(addFlow)

    else: #append the first flow in the list
      addFlow=[src_ip, dest_ip, src_port, dest_port, proto, 1, 1, 0, pkt_bytes, pkt_bytes, 0, int(time.time()), int(time.time()), 0]
      flowList.append(addFlow)
  except:
    pass

  results.append(str(( x.sprintf("{IP:%IP.src%,%IP.dst%,%IP.len%,}"
      "{TCP:tcp,%TCP.sport%,%TCP.dport%}"
      "{UDP:udp,%UDP.sport%,%UDP.dport%}"))).split(","))




def main():
  results = []
  flowList = []
  print("gathering data")
  pkts = sniff(filter="not port ssh and not port domain", prn = lambda x: fields_extraction(x, results, flowList), count = 10000)

  F = open('test.csv', 'w') 
  temp = str(flowList)
  temp = temp.replace("],", '\n')
  temp = temp.replace("[[", '')
  temp = temp.replace("]]", '')
  temp = temp.replace("[", '')
  temp = temp.replace("]", '')
  temp = temp.replace("'", '')
  temp = temp.replace(" ", '')
 #F.write("Flow ID, src_ip, dest_ip, src_port, dest_port, proto, tot pkts, src pkts, dest pkts, total bytes, src bytes, dest bytes, label \n")
  F.write(temp)
  '''
  I guess after the pkts are done counting, start to write them to a .csv file
  Depending on how the .csv process is:
    If it reads line by line then should be able to skip any flow that has less than 100 pkts

    If it imports the entire flowList all at once, then need to parse through the flowList and
    eliminate the flows with less than 100 total pkts. If this is the case then we need to 
    overwrite the flowIDs to make them more consistant
  '''

  x=4  #debuggin purpose only

main()

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

