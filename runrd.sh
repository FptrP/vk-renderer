#!/usr/bin/bash

case $1 in
  clean)
    rm captures/*
    exit;;
esac

renderdoccmd capture --opt-api-validation -c captures/debug ./main