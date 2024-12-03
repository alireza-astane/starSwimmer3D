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
    sphere { m*<-0.722200724015398,-1.0933523938852592,-0.5840061693164149>, 1 }        
    sphere {  m*<0.6969667701847645,-0.10341348000534145,9.265283927718738>, 1 }
    sphere {  m*<8.06475396850756,-0.3885057307976042,-5.305393501355196>, 1 }
    sphere {  m*<-6.831209225181428,6.134575642823052,-3.814586598173592>, 1}
    sphere { m*<-2.4905149907018442,-4.944395589547773,-1.4028904895940917>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6969667701847645,-0.10341348000534145,9.265283927718738>, <-0.722200724015398,-1.0933523938852592,-0.5840061693164149>, 0.5 }
    cylinder { m*<8.06475396850756,-0.3885057307976042,-5.305393501355196>, <-0.722200724015398,-1.0933523938852592,-0.5840061693164149>, 0.5}
    cylinder { m*<-6.831209225181428,6.134575642823052,-3.814586598173592>, <-0.722200724015398,-1.0933523938852592,-0.5840061693164149>, 0.5 }
    cylinder {  m*<-2.4905149907018442,-4.944395589547773,-1.4028904895940917>, <-0.722200724015398,-1.0933523938852592,-0.5840061693164149>, 0.5}

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
    sphere { m*<-0.722200724015398,-1.0933523938852592,-0.5840061693164149>, 1 }        
    sphere {  m*<0.6969667701847645,-0.10341348000534145,9.265283927718738>, 1 }
    sphere {  m*<8.06475396850756,-0.3885057307976042,-5.305393501355196>, 1 }
    sphere {  m*<-6.831209225181428,6.134575642823052,-3.814586598173592>, 1}
    sphere { m*<-2.4905149907018442,-4.944395589547773,-1.4028904895940917>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6969667701847645,-0.10341348000534145,9.265283927718738>, <-0.722200724015398,-1.0933523938852592,-0.5840061693164149>, 0.5 }
    cylinder { m*<8.06475396850756,-0.3885057307976042,-5.305393501355196>, <-0.722200724015398,-1.0933523938852592,-0.5840061693164149>, 0.5}
    cylinder { m*<-6.831209225181428,6.134575642823052,-3.814586598173592>, <-0.722200724015398,-1.0933523938852592,-0.5840061693164149>, 0.5 }
    cylinder {  m*<-2.4905149907018442,-4.944395589547773,-1.4028904895940917>, <-0.722200724015398,-1.0933523938852592,-0.5840061693164149>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    