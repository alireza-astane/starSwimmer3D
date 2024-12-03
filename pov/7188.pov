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
    sphere { m*<-0.7217423080831907,-1.0923540534942051,-0.5837938826057764>, 1 }        
    sphere {  m*<0.6974251861169717,-0.10241513961428739,9.265496214429376>, 1 }
    sphere {  m*<8.065212384439768,-0.38750739040655047,-5.305181214644557>, 1 }
    sphere {  m*<-6.830750809249221,6.135573983214106,-3.814374311462953>, 1}
    sphere { m*<-2.4929245543495275,-4.9496431486862935,-1.4040063285187507>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6974251861169717,-0.10241513961428739,9.265496214429376>, <-0.7217423080831907,-1.0923540534942051,-0.5837938826057764>, 0.5 }
    cylinder { m*<8.065212384439768,-0.38750739040655047,-5.305181214644557>, <-0.7217423080831907,-1.0923540534942051,-0.5837938826057764>, 0.5}
    cylinder { m*<-6.830750809249221,6.135573983214106,-3.814374311462953>, <-0.7217423080831907,-1.0923540534942051,-0.5837938826057764>, 0.5 }
    cylinder {  m*<-2.4929245543495275,-4.9496431486862935,-1.4040063285187507>, <-0.7217423080831907,-1.0923540534942051,-0.5837938826057764>, 0.5}

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
    sphere { m*<-0.7217423080831907,-1.0923540534942051,-0.5837938826057764>, 1 }        
    sphere {  m*<0.6974251861169717,-0.10241513961428739,9.265496214429376>, 1 }
    sphere {  m*<8.065212384439768,-0.38750739040655047,-5.305181214644557>, 1 }
    sphere {  m*<-6.830750809249221,6.135573983214106,-3.814374311462953>, 1}
    sphere { m*<-2.4929245543495275,-4.9496431486862935,-1.4040063285187507>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6974251861169717,-0.10241513961428739,9.265496214429376>, <-0.7217423080831907,-1.0923540534942051,-0.5837938826057764>, 0.5 }
    cylinder { m*<8.065212384439768,-0.38750739040655047,-5.305181214644557>, <-0.7217423080831907,-1.0923540534942051,-0.5837938826057764>, 0.5}
    cylinder { m*<-6.830750809249221,6.135573983214106,-3.814374311462953>, <-0.7217423080831907,-1.0923540534942051,-0.5837938826057764>, 0.5 }
    cylinder {  m*<-2.4929245543495275,-4.9496431486862935,-1.4040063285187507>, <-0.7217423080831907,-1.0923540534942051,-0.5837938826057764>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    