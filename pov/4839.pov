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
    sphere { m*<-0.24605063082144268,-0.12676984659182428,-1.4092220430944442>, 1 }        
    sphere {  m*<0.46652527921396325,0.25421198482193785,7.433940079422373>, 1 }
    sphere {  m*<2.4886577631848144,-0.02473587120545008,-2.6384315685456254>, 1 }
    sphere {  m*<-1.8676659907143327,2.2017040978267746,-2.383167808510412>, 1}
    sphere { m*<-1.599878769676501,-2.685987844577123,-2.1936215233478396>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.46652527921396325,0.25421198482193785,7.433940079422373>, <-0.24605063082144268,-0.12676984659182428,-1.4092220430944442>, 0.5 }
    cylinder { m*<2.4886577631848144,-0.02473587120545008,-2.6384315685456254>, <-0.24605063082144268,-0.12676984659182428,-1.4092220430944442>, 0.5}
    cylinder { m*<-1.8676659907143327,2.2017040978267746,-2.383167808510412>, <-0.24605063082144268,-0.12676984659182428,-1.4092220430944442>, 0.5 }
    cylinder {  m*<-1.599878769676501,-2.685987844577123,-2.1936215233478396>, <-0.24605063082144268,-0.12676984659182428,-1.4092220430944442>, 0.5}

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
    sphere { m*<-0.24605063082144268,-0.12676984659182428,-1.4092220430944442>, 1 }        
    sphere {  m*<0.46652527921396325,0.25421198482193785,7.433940079422373>, 1 }
    sphere {  m*<2.4886577631848144,-0.02473587120545008,-2.6384315685456254>, 1 }
    sphere {  m*<-1.8676659907143327,2.2017040978267746,-2.383167808510412>, 1}
    sphere { m*<-1.599878769676501,-2.685987844577123,-2.1936215233478396>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.46652527921396325,0.25421198482193785,7.433940079422373>, <-0.24605063082144268,-0.12676984659182428,-1.4092220430944442>, 0.5 }
    cylinder { m*<2.4886577631848144,-0.02473587120545008,-2.6384315685456254>, <-0.24605063082144268,-0.12676984659182428,-1.4092220430944442>, 0.5}
    cylinder { m*<-1.8676659907143327,2.2017040978267746,-2.383167808510412>, <-0.24605063082144268,-0.12676984659182428,-1.4092220430944442>, 0.5 }
    cylinder {  m*<-1.599878769676501,-2.685987844577123,-2.1936215233478396>, <-0.24605063082144268,-0.12676984659182428,-1.4092220430944442>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    