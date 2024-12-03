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
    sphere { m*<0.9905298775438791,0.48745311042449385,0.4515344701652867>, 1 }        
    sphere {  m*<1.2343862950552082,0.5271307397417634,3.4413421234722144>, 1 }
    sphere {  m*<3.727633484117746,0.5271307397417632,-0.7759400850184046>, 1 }
    sphere {  m*<-2.803334779844015,6.4267752525233455,-1.7916409373002617>, 1}
    sphere { m*<-3.816516996980861,-7.80762532633402,-2.3900286984862786>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.2343862950552082,0.5271307397417634,3.4413421234722144>, <0.9905298775438791,0.48745311042449385,0.4515344701652867>, 0.5 }
    cylinder { m*<3.727633484117746,0.5271307397417632,-0.7759400850184046>, <0.9905298775438791,0.48745311042449385,0.4515344701652867>, 0.5}
    cylinder { m*<-2.803334779844015,6.4267752525233455,-1.7916409373002617>, <0.9905298775438791,0.48745311042449385,0.4515344701652867>, 0.5 }
    cylinder {  m*<-3.816516996980861,-7.80762532633402,-2.3900286984862786>, <0.9905298775438791,0.48745311042449385,0.4515344701652867>, 0.5}

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
    sphere { m*<0.9905298775438791,0.48745311042449385,0.4515344701652867>, 1 }        
    sphere {  m*<1.2343862950552082,0.5271307397417634,3.4413421234722144>, 1 }
    sphere {  m*<3.727633484117746,0.5271307397417632,-0.7759400850184046>, 1 }
    sphere {  m*<-2.803334779844015,6.4267752525233455,-1.7916409373002617>, 1}
    sphere { m*<-3.816516996980861,-7.80762532633402,-2.3900286984862786>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.2343862950552082,0.5271307397417634,3.4413421234722144>, <0.9905298775438791,0.48745311042449385,0.4515344701652867>, 0.5 }
    cylinder { m*<3.727633484117746,0.5271307397417632,-0.7759400850184046>, <0.9905298775438791,0.48745311042449385,0.4515344701652867>, 0.5}
    cylinder { m*<-2.803334779844015,6.4267752525233455,-1.7916409373002617>, <0.9905298775438791,0.48745311042449385,0.4515344701652867>, 0.5 }
    cylinder {  m*<-3.816516996980861,-7.80762532633402,-2.3900286984862786>, <0.9905298775438791,0.48745311042449385,0.4515344701652867>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    