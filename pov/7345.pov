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
    sphere { m*<-0.6479744787416948,-0.9317021249336275,-0.549632919293602>, 1 }        
    sphere {  m*<0.7711930154584675,0.05823678894629025,9.29965717774155>, 1 }
    sphere {  m*<8.138980213781263,-0.22685546184597238,-5.2710202513323825>, 1 }
    sphere {  m*<-6.756982979907721,6.296225911774672,-3.7802133481507774>, 1}
    sphere { m*<-2.8694295190507595,-5.769597446268695,-1.578361089948272>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7711930154584675,0.05823678894629025,9.29965717774155>, <-0.6479744787416948,-0.9317021249336275,-0.549632919293602>, 0.5 }
    cylinder { m*<8.138980213781263,-0.22685546184597238,-5.2710202513323825>, <-0.6479744787416948,-0.9317021249336275,-0.549632919293602>, 0.5}
    cylinder { m*<-6.756982979907721,6.296225911774672,-3.7802133481507774>, <-0.6479744787416948,-0.9317021249336275,-0.549632919293602>, 0.5 }
    cylinder {  m*<-2.8694295190507595,-5.769597446268695,-1.578361089948272>, <-0.6479744787416948,-0.9317021249336275,-0.549632919293602>, 0.5}

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
    sphere { m*<-0.6479744787416948,-0.9317021249336275,-0.549632919293602>, 1 }        
    sphere {  m*<0.7711930154584675,0.05823678894629025,9.29965717774155>, 1 }
    sphere {  m*<8.138980213781263,-0.22685546184597238,-5.2710202513323825>, 1 }
    sphere {  m*<-6.756982979907721,6.296225911774672,-3.7802133481507774>, 1}
    sphere { m*<-2.8694295190507595,-5.769597446268695,-1.578361089948272>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7711930154584675,0.05823678894629025,9.29965717774155>, <-0.6479744787416948,-0.9317021249336275,-0.549632919293602>, 0.5 }
    cylinder { m*<8.138980213781263,-0.22685546184597238,-5.2710202513323825>, <-0.6479744787416948,-0.9317021249336275,-0.549632919293602>, 0.5}
    cylinder { m*<-6.756982979907721,6.296225911774672,-3.7802133481507774>, <-0.6479744787416948,-0.9317021249336275,-0.549632919293602>, 0.5 }
    cylinder {  m*<-2.8694295190507595,-5.769597446268695,-1.578361089948272>, <-0.6479744787416948,-0.9317021249336275,-0.549632919293602>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    