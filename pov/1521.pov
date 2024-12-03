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
    sphere { m*<0.7098099763524551,-5.46797478673208e-18,0.9184657183015346>, 1 }        
    sphere {  m*<0.8207329838073126,-1.4772884694555007e-18,3.9164183375409802>, 1 }
    sphere {  m*<6.610508339440313,3.3116880367624354e-18,-1.4168384574642447>, 1 }
    sphere {  m*<-4.124265869131128,8.164965809277259,-2.2382940569402185>, 1}
    sphere { m*<-4.124265869131128,-8.164965809277259,-2.238294056940221>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8207329838073126,-1.4772884694555007e-18,3.9164183375409802>, <0.7098099763524551,-5.46797478673208e-18,0.9184657183015346>, 0.5 }
    cylinder { m*<6.610508339440313,3.3116880367624354e-18,-1.4168384574642447>, <0.7098099763524551,-5.46797478673208e-18,0.9184657183015346>, 0.5}
    cylinder { m*<-4.124265869131128,8.164965809277259,-2.2382940569402185>, <0.7098099763524551,-5.46797478673208e-18,0.9184657183015346>, 0.5 }
    cylinder {  m*<-4.124265869131128,-8.164965809277259,-2.238294056940221>, <0.7098099763524551,-5.46797478673208e-18,0.9184657183015346>, 0.5}

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
    sphere { m*<0.7098099763524551,-5.46797478673208e-18,0.9184657183015346>, 1 }        
    sphere {  m*<0.8207329838073126,-1.4772884694555007e-18,3.9164183375409802>, 1 }
    sphere {  m*<6.610508339440313,3.3116880367624354e-18,-1.4168384574642447>, 1 }
    sphere {  m*<-4.124265869131128,8.164965809277259,-2.2382940569402185>, 1}
    sphere { m*<-4.124265869131128,-8.164965809277259,-2.238294056940221>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8207329838073126,-1.4772884694555007e-18,3.9164183375409802>, <0.7098099763524551,-5.46797478673208e-18,0.9184657183015346>, 0.5 }
    cylinder { m*<6.610508339440313,3.3116880367624354e-18,-1.4168384574642447>, <0.7098099763524551,-5.46797478673208e-18,0.9184657183015346>, 0.5}
    cylinder { m*<-4.124265869131128,8.164965809277259,-2.2382940569402185>, <0.7098099763524551,-5.46797478673208e-18,0.9184657183015346>, 0.5 }
    cylinder {  m*<-4.124265869131128,-8.164965809277259,-2.238294056940221>, <0.7098099763524551,-5.46797478673208e-18,0.9184657183015346>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    