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
    sphere { m*<-0.5871076277511122,-0.7991460082044904,-0.5214462384304408>, 1 }        
    sphere {  m*<0.8320598664490498,0.19079290567542695,9.327843858604709>, 1 }
    sphere {  m*<8.19984706477185,-0.09429934511683458,-5.242833570469221>, 1 }
    sphere {  m*<-6.696116128917139,6.428782028503804,-3.7520266672876152>, 1}
    sphere { m*<-3.1670601155464246,-6.417778770736199,-1.7161901134972903>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8320598664490498,0.19079290567542695,9.327843858604709>, <-0.5871076277511122,-0.7991460082044904,-0.5214462384304408>, 0.5 }
    cylinder { m*<8.19984706477185,-0.09429934511683458,-5.242833570469221>, <-0.5871076277511122,-0.7991460082044904,-0.5214462384304408>, 0.5}
    cylinder { m*<-6.696116128917139,6.428782028503804,-3.7520266672876152>, <-0.5871076277511122,-0.7991460082044904,-0.5214462384304408>, 0.5 }
    cylinder {  m*<-3.1670601155464246,-6.417778770736199,-1.7161901134972903>, <-0.5871076277511122,-0.7991460082044904,-0.5214462384304408>, 0.5}

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
    sphere { m*<-0.5871076277511122,-0.7991460082044904,-0.5214462384304408>, 1 }        
    sphere {  m*<0.8320598664490498,0.19079290567542695,9.327843858604709>, 1 }
    sphere {  m*<8.19984706477185,-0.09429934511683458,-5.242833570469221>, 1 }
    sphere {  m*<-6.696116128917139,6.428782028503804,-3.7520266672876152>, 1}
    sphere { m*<-3.1670601155464246,-6.417778770736199,-1.7161901134972903>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8320598664490498,0.19079290567542695,9.327843858604709>, <-0.5871076277511122,-0.7991460082044904,-0.5214462384304408>, 0.5 }
    cylinder { m*<8.19984706477185,-0.09429934511683458,-5.242833570469221>, <-0.5871076277511122,-0.7991460082044904,-0.5214462384304408>, 0.5}
    cylinder { m*<-6.696116128917139,6.428782028503804,-3.7520266672876152>, <-0.5871076277511122,-0.7991460082044904,-0.5214462384304408>, 0.5 }
    cylinder {  m*<-3.1670601155464246,-6.417778770736199,-1.7161901134972903>, <-0.5871076277511122,-0.7991460082044904,-0.5214462384304408>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    