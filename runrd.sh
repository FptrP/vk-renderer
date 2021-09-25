#!/usr/bin/bash

case $1 in
  clean)
    rm captures/*
    exit;;
esac

renderdoccmd capture -c captures/debug ./main