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
    sphere { m*<0.4663500038325798,-5.207548339947822e-18,1.0218391428709233>, 1 }        
    sphere {  m*<0.5337127909881662,-9.044108402408467e-19,4.021085005111948>, 1 }
    sphere {  m*<7.604759220251372,3.1821705332899815e-18,-1.683929646182319>, 1 }
    sphere {  m*<-4.321357732198796,8.164965809277259,-2.204805851625018>, 1}
    sphere { m*<-4.321357732198796,-8.164965809277259,-2.2048058516250215>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5337127909881662,-9.044108402408467e-19,4.021085005111948>, <0.4663500038325798,-5.207548339947822e-18,1.0218391428709233>, 0.5 }
    cylinder { m*<7.604759220251372,3.1821705332899815e-18,-1.683929646182319>, <0.4663500038325798,-5.207548339947822e-18,1.0218391428709233>, 0.5}
    cylinder { m*<-4.321357732198796,8.164965809277259,-2.204805851625018>, <0.4663500038325798,-5.207548339947822e-18,1.0218391428709233>, 0.5 }
    cylinder {  m*<-4.321357732198796,-8.164965809277259,-2.2048058516250215>, <0.4663500038325798,-5.207548339947822e-18,1.0218391428709233>, 0.5}

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
    sphere { m*<0.4663500038325798,-5.207548339947822e-18,1.0218391428709233>, 1 }        
    sphere {  m*<0.5337127909881662,-9.044108402408467e-19,4.021085005111948>, 1 }
    sphere {  m*<7.604759220251372,3.1821705332899815e-18,-1.683929646182319>, 1 }
    sphere {  m*<-4.321357732198796,8.164965809277259,-2.204805851625018>, 1}
    sphere { m*<-4.321357732198796,-8.164965809277259,-2.2048058516250215>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5337127909881662,-9.044108402408467e-19,4.021085005111948>, <0.4663500038325798,-5.207548339947822e-18,1.0218391428709233>, 0.5 }
    cylinder { m*<7.604759220251372,3.1821705332899815e-18,-1.683929646182319>, <0.4663500038325798,-5.207548339947822e-18,1.0218391428709233>, 0.5}
    cylinder { m*<-4.321357732198796,8.164965809277259,-2.204805851625018>, <0.4663500038325798,-5.207548339947822e-18,1.0218391428709233>, 0.5 }
    cylinder {  m*<-4.321357732198796,-8.164965809277259,-2.2048058516250215>, <0.4663500038325798,-5.207548339947822e-18,1.0218391428709233>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    