#Program:   Scapy_skeleton.py
#Authors:   Daniel Weigle, David Mercado, Jared Rigdon
#Purpose:   Collects a series of flows for various types of online activities and stores them into a .csv
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


#used to find the average of the of the features
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

#after all of the flows are captured, this function will find the avg of each feature by the lable
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
          flowAverage(flowList, "w")
          p=p+1
        else:
          flowAverage(flowList, "a")
        i=i+1
        keep_coll=0
        flowList = []
        
  getAvg()

main()