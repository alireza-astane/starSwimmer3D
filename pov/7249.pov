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
    sphere { m*<-0.6934866027294802,-1.030818644381122,-0.5707090172577763>, 1 }        
    sphere {  m*<0.7256808914706822,-0.0408797305012043,9.278581079777377>, 1 }
    sphere {  m*<8.093468089793475,-0.32597198129346716,-5.292096349296557>, 1 }
    sphere {  m*<-6.802495103895507,6.1971093923271825,-3.8012894461149527>, 1}
    sphere { m*<-2.6396156033691054,-5.26910761087717,-1.4719371269238937>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7256808914706822,-0.0408797305012043,9.278581079777377>, <-0.6934866027294802,-1.030818644381122,-0.5707090172577763>, 0.5 }
    cylinder { m*<8.093468089793475,-0.32597198129346716,-5.292096349296557>, <-0.6934866027294802,-1.030818644381122,-0.5707090172577763>, 0.5}
    cylinder { m*<-6.802495103895507,6.1971093923271825,-3.8012894461149527>, <-0.6934866027294802,-1.030818644381122,-0.5707090172577763>, 0.5 }
    cylinder {  m*<-2.6396156033691054,-5.26910761087717,-1.4719371269238937>, <-0.6934866027294802,-1.030818644381122,-0.5707090172577763>, 0.5}

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
    sphere { m*<-0.6934866027294802,-1.030818644381122,-0.5707090172577763>, 1 }        
    sphere {  m*<0.7256808914706822,-0.0408797305012043,9.278581079777377>, 1 }
    sphere {  m*<8.093468089793475,-0.32597198129346716,-5.292096349296557>, 1 }
    sphere {  m*<-6.802495103895507,6.1971093923271825,-3.8012894461149527>, 1}
    sphere { m*<-2.6396156033691054,-5.26910761087717,-1.4719371269238937>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7256808914706822,-0.0408797305012043,9.278581079777377>, <-0.6934866027294802,-1.030818644381122,-0.5707090172577763>, 0.5 }
    cylinder { m*<8.093468089793475,-0.32597198129346716,-5.292096349296557>, <-0.6934866027294802,-1.030818644381122,-0.5707090172577763>, 0.5}
    cylinder { m*<-6.802495103895507,6.1971093923271825,-3.8012894461149527>, <-0.6934866027294802,-1.030818644381122,-0.5707090172577763>, 0.5 }
    cylinder {  m*<-2.6396156033691054,-5.26910761087717,-1.4719371269238937>, <-0.6934866027294802,-1.030818644381122,-0.5707090172577763>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    