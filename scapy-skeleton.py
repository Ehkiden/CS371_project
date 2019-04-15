from scapy.all import sniff
import pandas as pd
import numpy as np
import sys
import socket 
import os
import time
import csv

# extracts the fields and applies them to the array
def fields_extraction(x, flowList, flowLabel):

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
        if( src_ip == flow[0] and dest_ip == flow[1] and src_port == flow[2] and dest_port == flow[3] and proto == flow[4] ):
          #means we sent the pkt
          #update the tuple bytes and pkts
          flow[5] = flow[5] + 1           #total pkts
          flow[6] = flow[6] + 1           #src pkts
          flow[8] = flow[8] + pkt_bytes   #total bytes
          flow[9] = flow[9] + pkt_bytes #src bytes
          flow[12] = int(time.time())-flow[11]  #increase the duration
          flow_match=1


        elif( src_ip == flow[1] and dest_ip == flow[0] and src_port == flow[3] and dest_port == flow[2] and proto == flow[4] ):
          #means the we recieved the pkt
          #update the tuple bytes and pkts
          flow[5] = flow[5] + 1           #total pkts
          flow[7] = flow[7] + 1           #dest pkts
          flow[8] = flow[8] + pkt_bytes   #total bytes
          flow[10] = flow[10] + pkt_bytes #dest bytes
          flow[12] = int(time.time())-flow[11]  #increase the duration
          flow_match=1

      if(flow_match==0): #add a new flow id to the list
        addFlow=[src_ip, dest_ip, src_port, dest_port, proto, 1, 1, 0, pkt_bytes, pkt_bytes, 0, int(time.time()), int(time.time()), flowLabel]
        flowList.append(addFlow)
        print(len(flowList))

    else: #append the first flow in the list
      addFlow=[src_ip, dest_ip, src_port, dest_port, proto, 1, 1, 0, pkt_bytes, pkt_bytes, 0, int(time.time()), int(time.time()), flowLabel]
      flowList.append(addFlow)
  except:
    pass


#check the flow list and write/append to csv
def flowChecker(flowSent, action):
  #parse flowList and pop any flows that are less than 100
  flowGreat = []
  
  for flow in flowSent:
    if(flow[5]>99):
      flowGreat.append(flow)
  #action is whether to write or append
  F = open('flowData.csv', action) 
  #convert entire flow list to str and replace brackets with commas
  temp = str(flowGreat)
  temp = temp.replace("],", '\n')
  temp = temp.replace("[[", '')
  temp = temp.replace("]]", '\n')
  temp = temp.replace("[", '')
  temp = temp.replace("]", '')
  temp = temp.replace("'", '')
  temp = temp.replace(" ", '')
  F.write(temp)
  F.close()

def flowAverage(tempList, action):
  flow5Tot  = 0
  flow6Tot  = 0
  flow7Tot  = 0
  flow8Tot  = 0
  flow9Tot  = 0
  flow10Tot = 0
  flow12Tot = 0
  for flow in tempList:
    flow5Tot += float(flow[5])
    flow6Tot += float(flow[6])
    flow7Tot += float(flow[7])
    flow8Tot += float(flow[8])
    flow9Tot += float(flow[9])
    flow10Tot += float(flow[10])
    flow12Tot += float(flow[12])
  
  
  flow5Avg = int(flow5Tot/len(tempList))
  flow6Avg = int(flow6Tot/len(tempList))
  flow7Avg = int(flow7Tot/len(tempList))
  flow8Avg = int(flow8Tot/len(tempList))
  flow9Avg = int(flow9Tot/len(tempList))
  flow10Avg = int(flow10Tot/len(tempList))
  flow12Avg = int(flow12Tot/len(tempList))

  F2 = open('features.csv', action)
  write = ("%s, %s, %s, %s, %s, %s, %s\n")%(flow5Avg, flow6Avg, flow7Avg, flow8Avg, flow9Avg, flow10Avg, flow12Avg)

  F2.write(write)
  F2.close()

#check if the current flow list meets the sample size requirements
def filtered_flowCheck(flowList, curr_flowLabel):
  df = pd.read_csv('flowData.csv')
  columns_list = ['srcIP', 'dstIP', 'srcPort', 'destPort', 'proto', 'totalPkts', 'srcPkts', 'destPkts', 'totalBytes', 'srcBytes', 'destBytes', 'currTime', 'durrTime', 'label']
  df.columns = columns_list
  num = (df['label']==curr_flowLabel).sum()
  return num

def getAvg():
  df = pd.read_csv('flowData.csv')
  columns_list = ['srcIP', 'dstIP', 'srcPort', 'destPort', 'proto', 'totalPkts', 'srcPkts', 'destPkts', 'totalBytes', 'srcBytes', 'destBytes', 'currTime', 'durrTime', 'label']
  df.columns = columns_list

  avg_totPkts = df.groupby('label')['totalPkts'].mean()
  avg_srcPkts = df.groupby('label')['srcPkts'].mean()
  avg_destPkts = df.groupby('label')['destPkts'].mean()
  avg_totBytes = df.groupby('label')['totalBytes'].mean()
  avg_srcBytes = df.groupby('label')['srcBytes'].mean()
  avg_destBytes = df.groupby('label')['destBytes'].mean()
  avg_durrtime = df.groupby('label')['durrTime'].mean()

  with open('features.csv', 'w') as f:
    avg_totPkts.to_csv(f, header=False, index=False)

  with open('features.csv', 'a') as f:
    avg_srcPkts.to_csv(f, header=False, index=False)
    avg_destPkts.to_csv(f, header=False, index=False)
    avg_totBytes.to_csv(f, header=False, index=False)
    avg_srcBytes.to_csv(f, header=False, index=False)
    avg_destBytes.to_csv(f, header=False, index=False)
    avg_durrtime.to_csv(f, header=False, index=False)


def main():
  #list of each label type
  label_list = ["Web browsing","VideosStreaming","Video Conferencing","File Downloading"]
  #length of label list
  label_len=(len(label_list)+1)

  flowList = []

  #iterate through each label to get atleast 25 samples
  i=1
  y=0
  p=0
  #stop = 0
  keep_coll=0
  while(i<label_len):
    #change what user_input is based on whether we are still collecting or not
    if(keep_coll == 0):
      user_input = input("Ready for "+label_list[i-1]+"?")  #moving on
    else:
      user_input=1  #still collecting

    if(user_input):   #do not increment until user input
      #while the len of the
      while((len(flowList)<24)):
        keep_coll=1
        print("gathering data")
        pkts = sniff(filter="not port ssh and not port domain", prn = lambda x: fields_extraction(x, flowList, i), count = 3000)
        
#      only append the data once we have 25+ flows of current activity
      if(y==0):
        y=y+1
        flowChecker(flowList, "w")  #for creating and writing the csv
    
      else:
        flowChecker(flowList, "a")  #for appending and not overwriting instead
    
    
      #empty out the list array for next flow type
      #check if the filtered csv is currently at >= 25 then empty else keep going
      if(filtered_flowCheck(flowList, i)>24):
        if(p==0):
#          flowChecker(flowList, "w")  #for creating and writing the csv
          flowAverage(flowList, "w")
          p=p+1
        else:
#          sflowChecker(flowList, "a")  #for appending and not overwriting instead
          flowAverage(flowList, "a")
        i=i+1
        keep_coll=0
        flowList = []
        #stop = 1
        
      #else:
        #stop = 0
  x=4  #debuggin purpose only
  getAvg()

main()

'''
    if(j==0):
      #temp = flowData[flowData[14] == j]    #retrieve rows with associated label value
      fD2 = flowData.groupby(flowData.columns[5])[flowData.columns[14]==j].mean()


      #write/overwrite to csv
      with open('flow_feat.csv', 'w') as f:
        fD2.to_csv(f, header=False)
    else:


      #append to csv
      with open('flow_feat.csv', 'a') as f:
        fD2.to_csv(f, header=False)

'''


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

