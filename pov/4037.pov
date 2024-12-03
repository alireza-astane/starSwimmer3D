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
    sphere { m*<-0.1531103100102311,-0.07707889698748162,-0.2558202608037262>, 1 }        
    sphere {  m*<0.10896997401855701,0.06304347479149908,2.9966310333561794>, 1 }
    sphere {  m*<2.5815980839960257,0.024955078398892494,-1.48502978625491>, 1 }
    sphere {  m*<-1.7747256699031213,2.2513950474311173,-1.2297660262196966>, 1}
    sphere { m*<-1.5069384488652895,-2.63629689497278,-1.040219741057124>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.10896997401855701,0.06304347479149908,2.9966310333561794>, <-0.1531103100102311,-0.07707889698748162,-0.2558202608037262>, 0.5 }
    cylinder { m*<2.5815980839960257,0.024955078398892494,-1.48502978625491>, <-0.1531103100102311,-0.07707889698748162,-0.2558202608037262>, 0.5}
    cylinder { m*<-1.7747256699031213,2.2513950474311173,-1.2297660262196966>, <-0.1531103100102311,-0.07707889698748162,-0.2558202608037262>, 0.5 }
    cylinder {  m*<-1.5069384488652895,-2.63629689497278,-1.040219741057124>, <-0.1531103100102311,-0.07707889698748162,-0.2558202608037262>, 0.5}

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
    sphere { m*<-0.1531103100102311,-0.07707889698748162,-0.2558202608037262>, 1 }        
    sphere {  m*<0.10896997401855701,0.06304347479149908,2.9966310333561794>, 1 }
    sphere {  m*<2.5815980839960257,0.024955078398892494,-1.48502978625491>, 1 }
    sphere {  m*<-1.7747256699031213,2.2513950474311173,-1.2297660262196966>, 1}
    sphere { m*<-1.5069384488652895,-2.63629689497278,-1.040219741057124>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.10896997401855701,0.06304347479149908,2.9966310333561794>, <-0.1531103100102311,-0.07707889698748162,-0.2558202608037262>, 0.5 }
    cylinder { m*<2.5815980839960257,0.024955078398892494,-1.48502978625491>, <-0.1531103100102311,-0.07707889698748162,-0.2558202608037262>, 0.5}
    cylinder { m*<-1.7747256699031213,2.2513950474311173,-1.2297660262196966>, <-0.1531103100102311,-0.07707889698748162,-0.2558202608037262>, 0.5 }
    cylinder {  m*<-1.5069384488652895,-2.63629689497278,-1.040219741057124>, <-0.1531103100102311,-0.07707889698748162,-0.2558202608037262>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    