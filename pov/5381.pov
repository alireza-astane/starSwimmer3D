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
    sphere { m*<-0.7390503679183364,-0.15624102030659498,-1.4740396194417862>, 1 }        
    sphere {  m*<0.33066085150080166,0.2864136380652332,8.458706816531087>, 1 }
    sphere {  m*<4.369294908291039,0.02790896400485049,-3.936837427940646>, 1 }
    sphere {  m*<-2.383833676261924,2.1725280577580928,-2.407589803905729>, 1}
    sphere { m*<-2.116046455224092,-2.7151638846458046,-2.2180435187431584>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.33066085150080166,0.2864136380652332,8.458706816531087>, <-0.7390503679183364,-0.15624102030659498,-1.4740396194417862>, 0.5 }
    cylinder { m*<4.369294908291039,0.02790896400485049,-3.936837427940646>, <-0.7390503679183364,-0.15624102030659498,-1.4740396194417862>, 0.5}
    cylinder { m*<-2.383833676261924,2.1725280577580928,-2.407589803905729>, <-0.7390503679183364,-0.15624102030659498,-1.4740396194417862>, 0.5 }
    cylinder {  m*<-2.116046455224092,-2.7151638846458046,-2.2180435187431584>, <-0.7390503679183364,-0.15624102030659498,-1.4740396194417862>, 0.5}

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
    sphere { m*<-0.7390503679183364,-0.15624102030659498,-1.4740396194417862>, 1 }        
    sphere {  m*<0.33066085150080166,0.2864136380652332,8.458706816531087>, 1 }
    sphere {  m*<4.369294908291039,0.02790896400485049,-3.936837427940646>, 1 }
    sphere {  m*<-2.383833676261924,2.1725280577580928,-2.407589803905729>, 1}
    sphere { m*<-2.116046455224092,-2.7151638846458046,-2.2180435187431584>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.33066085150080166,0.2864136380652332,8.458706816531087>, <-0.7390503679183364,-0.15624102030659498,-1.4740396194417862>, 0.5 }
    cylinder { m*<4.369294908291039,0.02790896400485049,-3.936837427940646>, <-0.7390503679183364,-0.15624102030659498,-1.4740396194417862>, 0.5}
    cylinder { m*<-2.383833676261924,2.1725280577580928,-2.407589803905729>, <-0.7390503679183364,-0.15624102030659498,-1.4740396194417862>, 0.5 }
    cylinder {  m*<-2.116046455224092,-2.7151638846458046,-2.2180435187431584>, <-0.7390503679183364,-0.15624102030659498,-1.4740396194417862>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    