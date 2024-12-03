#version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<-0.23143935867284576,-0.11895786593957242,-1.2278942061488518>, 1 }        
    sphere {  m*<0.4182244439900654,0.22838773170973165,6.834520269515136>, 1 }
    sphere {  m*<2.5032690353334113,-0.016923890553198172,-2.457103731600032>, 1 }
    sphere {  m*<-1.8530547185657356,2.2095160784790266,-2.201839971564819>, 1}
    sphere { m*<-1.5852674975279037,-2.6781758639248707,-2.012293686402246>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4182244439900654,0.22838773170973165,6.834520269515136>, <-0.23143935867284576,-0.11895786593957242,-1.2278942061488518>, 0.5 }
    cylinder { m*<2.5032690353334113,-0.016923890553198172,-2.457103731600032>, <-0.23143935867284576,-0.11895786593957242,-1.2278942061488518>, 0.5}
    cylinder { m*<-1.8530547185657356,2.2095160784790266,-2.201839971564819>, <-0.23143935867284576,-0.11895786593957242,-1.2278942061488518>, 0.5 }
    cylinder {  m*<-1.5852674975279037,-2.6781758639248707,-2.012293686402246>, <-0.23143935867284576,-0.11895786593957242,-1.2278942061488518>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    #version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<-0.23143935867284576,-0.11895786593957242,-1.2278942061488518>, 1 }        
    sphere {  m*<0.4182244439900654,0.22838773170973165,6.834520269515136>, 1 }
    sphere {  m*<2.5032690353334113,-0.016923890553198172,-2.457103731600032>, 1 }
    sphere {  m*<-1.8530547185657356,2.2095160784790266,-2.201839971564819>, 1}
    sphere { m*<-1.5852674975279037,-2.6781758639248707,-2.012293686402246>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4182244439900654,0.22838773170973165,6.834520269515136>, <-0.23143935867284576,-0.11895786593957242,-1.2278942061488518>, 0.5 }
    cylinder { m*<2.5032690353334113,-0.016923890553198172,-2.457103731600032>, <-0.23143935867284576,-0.11895786593957242,-1.2278942061488518>, 0.5}
    cylinder { m*<-1.8530547185657356,2.2095160784790266,-2.201839971564819>, <-0.23143935867284576,-0.11895786593957242,-1.2278942061488518>, 0.5 }
    cylinder {  m*<-1.5852674975279037,-2.6781758639248707,-2.012293686402246>, <-0.23143935867284576,-0.11895786593957242,-1.2278942061488518>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    