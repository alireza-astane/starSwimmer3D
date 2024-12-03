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
    sphere { m*<-1.2818325468637746,-0.17529702520974863,-1.1991532878033861>, 1 }        
    sphere {  m*<0.05416663991428905,0.2806488592676875,8.700671073530476>, 1 }
    sphere {  m*<6.227796601852915,0.0856940474209196,-5.071029665285678>, 1 }
    sphere {  m*<-2.948675483329045,2.153850056357683,-2.091732559825321>, 1}
    sphere { m*<-2.680888262291214,-2.7338418860462146,-1.9021862746627505>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.05416663991428905,0.2806488592676875,8.700671073530476>, <-1.2818325468637746,-0.17529702520974863,-1.1991532878033861>, 0.5 }
    cylinder { m*<6.227796601852915,0.0856940474209196,-5.071029665285678>, <-1.2818325468637746,-0.17529702520974863,-1.1991532878033861>, 0.5}
    cylinder { m*<-2.948675483329045,2.153850056357683,-2.091732559825321>, <-1.2818325468637746,-0.17529702520974863,-1.1991532878033861>, 0.5 }
    cylinder {  m*<-2.680888262291214,-2.7338418860462146,-1.9021862746627505>, <-1.2818325468637746,-0.17529702520974863,-1.1991532878033861>, 0.5}

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
    sphere { m*<-1.2818325468637746,-0.17529702520974863,-1.1991532878033861>, 1 }        
    sphere {  m*<0.05416663991428905,0.2806488592676875,8.700671073530476>, 1 }
    sphere {  m*<6.227796601852915,0.0856940474209196,-5.071029665285678>, 1 }
    sphere {  m*<-2.948675483329045,2.153850056357683,-2.091732559825321>, 1}
    sphere { m*<-2.680888262291214,-2.7338418860462146,-1.9021862746627505>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.05416663991428905,0.2806488592676875,8.700671073530476>, <-1.2818325468637746,-0.17529702520974863,-1.1991532878033861>, 0.5 }
    cylinder { m*<6.227796601852915,0.0856940474209196,-5.071029665285678>, <-1.2818325468637746,-0.17529702520974863,-1.1991532878033861>, 0.5}
    cylinder { m*<-2.948675483329045,2.153850056357683,-2.091732559825321>, <-1.2818325468637746,-0.17529702520974863,-1.1991532878033861>, 0.5 }
    cylinder {  m*<-2.680888262291214,-2.7338418860462146,-1.9021862746627505>, <-1.2818325468637746,-0.17529702520974863,-1.1991532878033861>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    