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
    sphere { m*<-0.4551718061509546,-0.14556188995662656,-1.5993784788335608>, 1 }        
    sphere {  m*<0.456732500824447,0.28908398580687944,8.349460706376098>, 1 }
    sphere {  m*<3.2881869192837394,-0.007812764421893742,-3.3315624610619214>, 1 }
    sphere {  m*<-2.086399276792543,2.183022765166415,-2.5568680639373182>, 1}
    sphere { m*<-1.818612055754711,-2.7046691772374825,-2.367321778774748>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.456732500824447,0.28908398580687944,8.349460706376098>, <-0.4551718061509546,-0.14556188995662656,-1.5993784788335608>, 0.5 }
    cylinder { m*<3.2881869192837394,-0.007812764421893742,-3.3315624610619214>, <-0.4551718061509546,-0.14556188995662656,-1.5993784788335608>, 0.5}
    cylinder { m*<-2.086399276792543,2.183022765166415,-2.5568680639373182>, <-0.4551718061509546,-0.14556188995662656,-1.5993784788335608>, 0.5 }
    cylinder {  m*<-1.818612055754711,-2.7046691772374825,-2.367321778774748>, <-0.4551718061509546,-0.14556188995662656,-1.5993784788335608>, 0.5}

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
    sphere { m*<-0.4551718061509546,-0.14556188995662656,-1.5993784788335608>, 1 }        
    sphere {  m*<0.456732500824447,0.28908398580687944,8.349460706376098>, 1 }
    sphere {  m*<3.2881869192837394,-0.007812764421893742,-3.3315624610619214>, 1 }
    sphere {  m*<-2.086399276792543,2.183022765166415,-2.5568680639373182>, 1}
    sphere { m*<-1.818612055754711,-2.7046691772374825,-2.367321778774748>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.456732500824447,0.28908398580687944,8.349460706376098>, <-0.4551718061509546,-0.14556188995662656,-1.5993784788335608>, 0.5 }
    cylinder { m*<3.2881869192837394,-0.007812764421893742,-3.3315624610619214>, <-0.4551718061509546,-0.14556188995662656,-1.5993784788335608>, 0.5}
    cylinder { m*<-2.086399276792543,2.183022765166415,-2.5568680639373182>, <-0.4551718061509546,-0.14556188995662656,-1.5993784788335608>, 0.5 }
    cylinder {  m*<-1.818612055754711,-2.7046691772374825,-2.367321778774748>, <-0.4551718061509546,-0.14556188995662656,-1.5993784788335608>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    