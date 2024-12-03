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
    sphere { m*<-1.3050213131311086,-0.6401456463121054,-0.8784931084663243>, 1 }        
    sphere {  m*<0.14437342168677336,0.015642512507819006,8.994236127558537>, 1 }
    sphere {  m*<7.499724859686746,-0.073277763486538,-5.585257162486812>, 1 }
    sphere {  m*<-4.789105066119582,3.808790471399812,-2.66347624814009>, 1}
    sphere { m*<-2.617336656602831,-3.259062054528977,-1.5245806319192858>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.14437342168677336,0.015642512507819006,8.994236127558537>, <-1.3050213131311086,-0.6401456463121054,-0.8784931084663243>, 0.5 }
    cylinder { m*<7.499724859686746,-0.073277763486538,-5.585257162486812>, <-1.3050213131311086,-0.6401456463121054,-0.8784931084663243>, 0.5}
    cylinder { m*<-4.789105066119582,3.808790471399812,-2.66347624814009>, <-1.3050213131311086,-0.6401456463121054,-0.8784931084663243>, 0.5 }
    cylinder {  m*<-2.617336656602831,-3.259062054528977,-1.5245806319192858>, <-1.3050213131311086,-0.6401456463121054,-0.8784931084663243>, 0.5}

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
    sphere { m*<-1.3050213131311086,-0.6401456463121054,-0.8784931084663243>, 1 }        
    sphere {  m*<0.14437342168677336,0.015642512507819006,8.994236127558537>, 1 }
    sphere {  m*<7.499724859686746,-0.073277763486538,-5.585257162486812>, 1 }
    sphere {  m*<-4.789105066119582,3.808790471399812,-2.66347624814009>, 1}
    sphere { m*<-2.617336656602831,-3.259062054528977,-1.5245806319192858>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.14437342168677336,0.015642512507819006,8.994236127558537>, <-1.3050213131311086,-0.6401456463121054,-0.8784931084663243>, 0.5 }
    cylinder { m*<7.499724859686746,-0.073277763486538,-5.585257162486812>, <-1.3050213131311086,-0.6401456463121054,-0.8784931084663243>, 0.5}
    cylinder { m*<-4.789105066119582,3.808790471399812,-2.66347624814009>, <-1.3050213131311086,-0.6401456463121054,-0.8784931084663243>, 0.5 }
    cylinder {  m*<-2.617336656602831,-3.259062054528977,-1.5245806319192858>, <-1.3050213131311086,-0.6401456463121054,-0.8784931084663243>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    