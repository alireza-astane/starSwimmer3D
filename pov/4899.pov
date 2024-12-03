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
    sphere { m*<-0.2540044974618952,-0.1310224162294224,-1.5079305893865926>, 1 }        
    sphere {  m*<0.49227432723734776,0.26797882612958535,7.753489201087104>, 1 }
    sphere {  m*<2.480703896544362,-0.028988440843048184,-2.7371401148377736>, 1 }
    sphere {  m*<-1.875619857354785,2.197451528189177,-2.48187635480256>, 1}
    sphere { m*<-1.6078326363169533,-2.6902404142147205,-2.2923300696399873>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.49227432723734776,0.26797882612958535,7.753489201087104>, <-0.2540044974618952,-0.1310224162294224,-1.5079305893865926>, 0.5 }
    cylinder { m*<2.480703896544362,-0.028988440843048184,-2.7371401148377736>, <-0.2540044974618952,-0.1310224162294224,-1.5079305893865926>, 0.5}
    cylinder { m*<-1.875619857354785,2.197451528189177,-2.48187635480256>, <-0.2540044974618952,-0.1310224162294224,-1.5079305893865926>, 0.5 }
    cylinder {  m*<-1.6078326363169533,-2.6902404142147205,-2.2923300696399873>, <-0.2540044974618952,-0.1310224162294224,-1.5079305893865926>, 0.5}

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
    sphere { m*<-0.2540044974618952,-0.1310224162294224,-1.5079305893865926>, 1 }        
    sphere {  m*<0.49227432723734776,0.26797882612958535,7.753489201087104>, 1 }
    sphere {  m*<2.480703896544362,-0.028988440843048184,-2.7371401148377736>, 1 }
    sphere {  m*<-1.875619857354785,2.197451528189177,-2.48187635480256>, 1}
    sphere { m*<-1.6078326363169533,-2.6902404142147205,-2.2923300696399873>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.49227432723734776,0.26797882612958535,7.753489201087104>, <-0.2540044974618952,-0.1310224162294224,-1.5079305893865926>, 0.5 }
    cylinder { m*<2.480703896544362,-0.028988440843048184,-2.7371401148377736>, <-0.2540044974618952,-0.1310224162294224,-1.5079305893865926>, 0.5}
    cylinder { m*<-1.875619857354785,2.197451528189177,-2.48187635480256>, <-0.2540044974618952,-0.1310224162294224,-1.5079305893865926>, 0.5 }
    cylinder {  m*<-1.6078326363169533,-2.6902404142147205,-2.2923300696399873>, <-0.2540044974618952,-0.1310224162294224,-1.5079305893865926>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    