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
    sphere { m*<-9.691489387175789e-05,0.2076951988007495,-0.1281066090295191>, 1 }        
    sphere {  m*<0.24063818984781998,0.33640527698107503,2.8594481620910326>, 1 }
    sphere {  m*<2.734611479112392,0.3097291741871243,-1.3573161344807056>, 1 }
    sphere {  m*<-1.6217122747867632,2.536169143219352,-1.102052374445491>, 1}
    sphere { m*<-2.209995713551601,-3.96980164657724,-1.4085080272964734>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.24063818984781998,0.33640527698107503,2.8594481620910326>, <-9.691489387175789e-05,0.2076951988007495,-0.1281066090295191>, 0.5 }
    cylinder { m*<2.734611479112392,0.3097291741871243,-1.3573161344807056>, <-9.691489387175789e-05,0.2076951988007495,-0.1281066090295191>, 0.5}
    cylinder { m*<-1.6217122747867632,2.536169143219352,-1.102052374445491>, <-9.691489387175789e-05,0.2076951988007495,-0.1281066090295191>, 0.5 }
    cylinder {  m*<-2.209995713551601,-3.96980164657724,-1.4085080272964734>, <-9.691489387175789e-05,0.2076951988007495,-0.1281066090295191>, 0.5}

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
    sphere { m*<-9.691489387175789e-05,0.2076951988007495,-0.1281066090295191>, 1 }        
    sphere {  m*<0.24063818984781998,0.33640527698107503,2.8594481620910326>, 1 }
    sphere {  m*<2.734611479112392,0.3097291741871243,-1.3573161344807056>, 1 }
    sphere {  m*<-1.6217122747867632,2.536169143219352,-1.102052374445491>, 1}
    sphere { m*<-2.209995713551601,-3.96980164657724,-1.4085080272964734>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.24063818984781998,0.33640527698107503,2.8594481620910326>, <-9.691489387175789e-05,0.2076951988007495,-0.1281066090295191>, 0.5 }
    cylinder { m*<2.734611479112392,0.3097291741871243,-1.3573161344807056>, <-9.691489387175789e-05,0.2076951988007495,-0.1281066090295191>, 0.5}
    cylinder { m*<-1.6217122747867632,2.536169143219352,-1.102052374445491>, <-9.691489387175789e-05,0.2076951988007495,-0.1281066090295191>, 0.5 }
    cylinder {  m*<-2.209995713551601,-3.96980164657724,-1.4085080272964734>, <-9.691489387175789e-05,0.2076951988007495,-0.1281066090295191>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    