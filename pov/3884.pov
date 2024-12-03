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
    sphere { m*<-0.09085157197004357,0.036136531105494196,-0.18068928462771466>, 1 }        
    sphere {  m*<0.149883532771648,0.1648466092858194,2.8068654864928355>, 1 }
    sphere {  m*<2.6438568220362195,0.13817050649186857,-1.4098988100789005>, 1 }
    sphere {  m*<-1.7124669318629349,2.3646104755240964,-1.154635050043686>, 1}
    sphere { m*<-1.8079569280845318,-3.209804963005846,-1.1755692920824306>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.149883532771648,0.1648466092858194,2.8068654864928355>, <-0.09085157197004357,0.036136531105494196,-0.18068928462771466>, 0.5 }
    cylinder { m*<2.6438568220362195,0.13817050649186857,-1.4098988100789005>, <-0.09085157197004357,0.036136531105494196,-0.18068928462771466>, 0.5}
    cylinder { m*<-1.7124669318629349,2.3646104755240964,-1.154635050043686>, <-0.09085157197004357,0.036136531105494196,-0.18068928462771466>, 0.5 }
    cylinder {  m*<-1.8079569280845318,-3.209804963005846,-1.1755692920824306>, <-0.09085157197004357,0.036136531105494196,-0.18068928462771466>, 0.5}

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
    sphere { m*<-0.09085157197004357,0.036136531105494196,-0.18068928462771466>, 1 }        
    sphere {  m*<0.149883532771648,0.1648466092858194,2.8068654864928355>, 1 }
    sphere {  m*<2.6438568220362195,0.13817050649186857,-1.4098988100789005>, 1 }
    sphere {  m*<-1.7124669318629349,2.3646104755240964,-1.154635050043686>, 1}
    sphere { m*<-1.8079569280845318,-3.209804963005846,-1.1755692920824306>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.149883532771648,0.1648466092858194,2.8068654864928355>, <-0.09085157197004357,0.036136531105494196,-0.18068928462771466>, 0.5 }
    cylinder { m*<2.6438568220362195,0.13817050649186857,-1.4098988100789005>, <-0.09085157197004357,0.036136531105494196,-0.18068928462771466>, 0.5}
    cylinder { m*<-1.7124669318629349,2.3646104755240964,-1.154635050043686>, <-0.09085157197004357,0.036136531105494196,-0.18068928462771466>, 0.5 }
    cylinder {  m*<-1.8079569280845318,-3.209804963005846,-1.1755692920824306>, <-0.09085157197004357,0.036136531105494196,-0.18068928462771466>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    