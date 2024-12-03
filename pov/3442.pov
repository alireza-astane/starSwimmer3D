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
    sphere { m*<0.18801463933575274,0.5632931224933244,-0.01911596159611348>, 1 }        
    sphere {  m*<0.4287497440774443,0.69200320067365,2.9684388095244367>, 1 }
    sphere {  m*<2.92272303334201,0.6653270978796988,-1.248325487047298>, 1 }
    sphere {  m*<-1.4336007205571377,2.8917670669119255,-0.9930617270120837>, 1}
    sphere { m*<-2.9253388039845936,-5.322055200206855,-1.8229732996854993>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4287497440774443,0.69200320067365,2.9684388095244367>, <0.18801463933575274,0.5632931224933244,-0.01911596159611348>, 0.5 }
    cylinder { m*<2.92272303334201,0.6653270978796988,-1.248325487047298>, <0.18801463933575274,0.5632931224933244,-0.01911596159611348>, 0.5}
    cylinder { m*<-1.4336007205571377,2.8917670669119255,-0.9930617270120837>, <0.18801463933575274,0.5632931224933244,-0.01911596159611348>, 0.5 }
    cylinder {  m*<-2.9253388039845936,-5.322055200206855,-1.8229732996854993>, <0.18801463933575274,0.5632931224933244,-0.01911596159611348>, 0.5}

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
    sphere { m*<0.18801463933575274,0.5632931224933244,-0.01911596159611348>, 1 }        
    sphere {  m*<0.4287497440774443,0.69200320067365,2.9684388095244367>, 1 }
    sphere {  m*<2.92272303334201,0.6653270978796988,-1.248325487047298>, 1 }
    sphere {  m*<-1.4336007205571377,2.8917670669119255,-0.9930617270120837>, 1}
    sphere { m*<-2.9253388039845936,-5.322055200206855,-1.8229732996854993>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4287497440774443,0.69200320067365,2.9684388095244367>, <0.18801463933575274,0.5632931224933244,-0.01911596159611348>, 0.5 }
    cylinder { m*<2.92272303334201,0.6653270978796988,-1.248325487047298>, <0.18801463933575274,0.5632931224933244,-0.01911596159611348>, 0.5}
    cylinder { m*<-1.4336007205571377,2.8917670669119255,-0.9930617270120837>, <0.18801463933575274,0.5632931224933244,-0.01911596159611348>, 0.5 }
    cylinder {  m*<-2.9253388039845936,-5.322055200206855,-1.8229732996854993>, <0.18801463933575274,0.5632931224933244,-0.01911596159611348>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    