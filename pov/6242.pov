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
    sphere { m*<-1.4350285161011411,-0.4469615701203236,-0.9452651038115132>, 1 }        
    sphere {  m*<0.02215830019243281,0.11771040408062128,8.93195221294553>, 1 }
    sphere {  m*<7.377509738192406,0.028790128086264077,-5.647541077099826>, 1 }
    sphere {  m*<-4.148523509555754,3.1310377890750316,-2.3361472292176377>, 1}
    sphere { m*<-2.7842067219985855,-3.042348305266411,-1.6100945305348573>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.02215830019243281,0.11771040408062128,8.93195221294553>, <-1.4350285161011411,-0.4469615701203236,-0.9452651038115132>, 0.5 }
    cylinder { m*<7.377509738192406,0.028790128086264077,-5.647541077099826>, <-1.4350285161011411,-0.4469615701203236,-0.9452651038115132>, 0.5}
    cylinder { m*<-4.148523509555754,3.1310377890750316,-2.3361472292176377>, <-1.4350285161011411,-0.4469615701203236,-0.9452651038115132>, 0.5 }
    cylinder {  m*<-2.7842067219985855,-3.042348305266411,-1.6100945305348573>, <-1.4350285161011411,-0.4469615701203236,-0.9452651038115132>, 0.5}

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
    sphere { m*<-1.4350285161011411,-0.4469615701203236,-0.9452651038115132>, 1 }        
    sphere {  m*<0.02215830019243281,0.11771040408062128,8.93195221294553>, 1 }
    sphere {  m*<7.377509738192406,0.028790128086264077,-5.647541077099826>, 1 }
    sphere {  m*<-4.148523509555754,3.1310377890750316,-2.3361472292176377>, 1}
    sphere { m*<-2.7842067219985855,-3.042348305266411,-1.6100945305348573>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.02215830019243281,0.11771040408062128,8.93195221294553>, <-1.4350285161011411,-0.4469615701203236,-0.9452651038115132>, 0.5 }
    cylinder { m*<7.377509738192406,0.028790128086264077,-5.647541077099826>, <-1.4350285161011411,-0.4469615701203236,-0.9452651038115132>, 0.5}
    cylinder { m*<-4.148523509555754,3.1310377890750316,-2.3361472292176377>, <-1.4350285161011411,-0.4469615701203236,-0.9452651038115132>, 0.5 }
    cylinder {  m*<-2.7842067219985855,-3.042348305266411,-1.6100945305348573>, <-1.4350285161011411,-0.4469615701203236,-0.9452651038115132>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    