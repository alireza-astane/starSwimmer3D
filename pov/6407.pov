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
    sphere { m*<-1.3127736066554414,-0.6291009868972219,-0.8824718022586073>, 1 }        
    sphere {  m*<0.1370812115604816,0.021701567490048174,8.990520022609324>, 1 }
    sphere {  m*<7.49243264956045,-0.06721870850430905,-5.588973267436025>, 1 }
    sphere {  m*<-4.752287885325751,3.770635878463449,-2.644668020096936>, 1}
    sphere { m*<-2.6271161306480937,-3.246745946460947,-1.5295898520661728>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.1370812115604816,0.021701567490048174,8.990520022609324>, <-1.3127736066554414,-0.6291009868972219,-0.8824718022586073>, 0.5 }
    cylinder { m*<7.49243264956045,-0.06721870850430905,-5.588973267436025>, <-1.3127736066554414,-0.6291009868972219,-0.8824718022586073>, 0.5}
    cylinder { m*<-4.752287885325751,3.770635878463449,-2.644668020096936>, <-1.3127736066554414,-0.6291009868972219,-0.8824718022586073>, 0.5 }
    cylinder {  m*<-2.6271161306480937,-3.246745946460947,-1.5295898520661728>, <-1.3127736066554414,-0.6291009868972219,-0.8824718022586073>, 0.5}

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
    sphere { m*<-1.3127736066554414,-0.6291009868972219,-0.8824718022586073>, 1 }        
    sphere {  m*<0.1370812115604816,0.021701567490048174,8.990520022609324>, 1 }
    sphere {  m*<7.49243264956045,-0.06721870850430905,-5.588973267436025>, 1 }
    sphere {  m*<-4.752287885325751,3.770635878463449,-2.644668020096936>, 1}
    sphere { m*<-2.6271161306480937,-3.246745946460947,-1.5295898520661728>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.1370812115604816,0.021701567490048174,8.990520022609324>, <-1.3127736066554414,-0.6291009868972219,-0.8824718022586073>, 0.5 }
    cylinder { m*<7.49243264956045,-0.06721870850430905,-5.588973267436025>, <-1.3127736066554414,-0.6291009868972219,-0.8824718022586073>, 0.5}
    cylinder { m*<-4.752287885325751,3.770635878463449,-2.644668020096936>, <-1.3127736066554414,-0.6291009868972219,-0.8824718022586073>, 0.5 }
    cylinder {  m*<-2.6271161306480937,-3.246745946460947,-1.5295898520661728>, <-1.3127736066554414,-0.6291009868972219,-0.8824718022586073>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    