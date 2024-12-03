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
    sphere { m*<-0.24263016587363007,-0.12494108002569171,-1.3667736166995776>, 1 }        
    sphere {  m*<0.45534114780744694,0.24823233997846417,7.295143516369264>, 1 }
    sphere {  m*<2.492078228132627,-0.022907104639317555,-2.5959831421507595>, 1 }
    sphere {  m*<-1.86424552576652,2.2035328643929075,-2.3407193821155463>, 1}
    sphere { m*<-1.5964583047286882,-2.68415907801099,-2.1511730969529737>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.45534114780744694,0.24823233997846417,7.295143516369264>, <-0.24263016587363007,-0.12494108002569171,-1.3667736166995776>, 0.5 }
    cylinder { m*<2.492078228132627,-0.022907104639317555,-2.5959831421507595>, <-0.24263016587363007,-0.12494108002569171,-1.3667736166995776>, 0.5}
    cylinder { m*<-1.86424552576652,2.2035328643929075,-2.3407193821155463>, <-0.24263016587363007,-0.12494108002569171,-1.3667736166995776>, 0.5 }
    cylinder {  m*<-1.5964583047286882,-2.68415907801099,-2.1511730969529737>, <-0.24263016587363007,-0.12494108002569171,-1.3667736166995776>, 0.5}

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
    sphere { m*<-0.24263016587363007,-0.12494108002569171,-1.3667736166995776>, 1 }        
    sphere {  m*<0.45534114780744694,0.24823233997846417,7.295143516369264>, 1 }
    sphere {  m*<2.492078228132627,-0.022907104639317555,-2.5959831421507595>, 1 }
    sphere {  m*<-1.86424552576652,2.2035328643929075,-2.3407193821155463>, 1}
    sphere { m*<-1.5964583047286882,-2.68415907801099,-2.1511730969529737>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.45534114780744694,0.24823233997846417,7.295143516369264>, <-0.24263016587363007,-0.12494108002569171,-1.3667736166995776>, 0.5 }
    cylinder { m*<2.492078228132627,-0.022907104639317555,-2.5959831421507595>, <-0.24263016587363007,-0.12494108002569171,-1.3667736166995776>, 0.5}
    cylinder { m*<-1.86424552576652,2.2035328643929075,-2.3407193821155463>, <-0.24263016587363007,-0.12494108002569171,-1.3667736166995776>, 0.5 }
    cylinder {  m*<-1.5964583047286882,-2.68415907801099,-2.1511730969529737>, <-0.24263016587363007,-0.12494108002569171,-1.3667736166995776>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    