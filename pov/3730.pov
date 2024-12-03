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
    sphere { m*<-0.0013241832516107666,0.20537522392407137,-0.12881768057125553>, 1 }        
    sphere {  m*<0.23941092149008103,0.3340853021043968,2.858737090549296>, 1 }
    sphere {  m*<2.733384210754653,0.3074091993104461,-1.3580272060224425>, 1 }
    sphere {  m*<-1.622939543144502,2.533849168342674,-1.1027634459872275>, 1}
    sphere { m*<-2.2049051172613496,-3.960178604129987,-1.4055585679303606>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.23941092149008103,0.3340853021043968,2.858737090549296>, <-0.0013241832516107666,0.20537522392407137,-0.12881768057125553>, 0.5 }
    cylinder { m*<2.733384210754653,0.3074091993104461,-1.3580272060224425>, <-0.0013241832516107666,0.20537522392407137,-0.12881768057125553>, 0.5}
    cylinder { m*<-1.622939543144502,2.533849168342674,-1.1027634459872275>, <-0.0013241832516107666,0.20537522392407137,-0.12881768057125553>, 0.5 }
    cylinder {  m*<-2.2049051172613496,-3.960178604129987,-1.4055585679303606>, <-0.0013241832516107666,0.20537522392407137,-0.12881768057125553>, 0.5}

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
    sphere { m*<-0.0013241832516107666,0.20537522392407137,-0.12881768057125553>, 1 }        
    sphere {  m*<0.23941092149008103,0.3340853021043968,2.858737090549296>, 1 }
    sphere {  m*<2.733384210754653,0.3074091993104461,-1.3580272060224425>, 1 }
    sphere {  m*<-1.622939543144502,2.533849168342674,-1.1027634459872275>, 1}
    sphere { m*<-2.2049051172613496,-3.960178604129987,-1.4055585679303606>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.23941092149008103,0.3340853021043968,2.858737090549296>, <-0.0013241832516107666,0.20537522392407137,-0.12881768057125553>, 0.5 }
    cylinder { m*<2.733384210754653,0.3074091993104461,-1.3580272060224425>, <-0.0013241832516107666,0.20537522392407137,-0.12881768057125553>, 0.5}
    cylinder { m*<-1.622939543144502,2.533849168342674,-1.1027634459872275>, <-0.0013241832516107666,0.20537522392407137,-0.12881768057125553>, 0.5 }
    cylinder {  m*<-2.2049051172613496,-3.960178604129987,-1.4055585679303606>, <-0.0013241832516107666,0.20537522392407137,-0.12881768057125553>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    