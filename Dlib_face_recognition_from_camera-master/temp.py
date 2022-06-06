# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 14:06:17 2021

@author: user
"""
import serial


ser = serial.Serial('COM4',9600)

def get_temp(self):
        temp=str(ser.readline())
        self.realtemp = str(temp[11:16])
        return self.realtemp