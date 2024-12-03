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
    sphere { m*<0.09681277307753526,-4.710574297700019e-18,1.1616954320169603>, 1 }        
    sphere {  m*<0.10932611123758684,-4.834832497323224e-18,4.1616697129296>, 1 }
    sphere {  m*<9.056207785767104,1.0991639745120106e-18,-2.0480637529838495>, 1 }
    sphere {  m*<-4.631095730252666,8.164965809277259,-2.151946510623188>, 1}
    sphere { m*<-4.631095730252666,-8.164965809277259,-2.1519465106231914>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.10932611123758684,-4.834832497323224e-18,4.1616697129296>, <0.09681277307753526,-4.710574297700019e-18,1.1616954320169603>, 0.5 }
    cylinder { m*<9.056207785767104,1.0991639745120106e-18,-2.0480637529838495>, <0.09681277307753526,-4.710574297700019e-18,1.1616954320169603>, 0.5}
    cylinder { m*<-4.631095730252666,8.164965809277259,-2.151946510623188>, <0.09681277307753526,-4.710574297700019e-18,1.1616954320169603>, 0.5 }
    cylinder {  m*<-4.631095730252666,-8.164965809277259,-2.1519465106231914>, <0.09681277307753526,-4.710574297700019e-18,1.1616954320169603>, 0.5}

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
    sphere { m*<0.09681277307753526,-4.710574297700019e-18,1.1616954320169603>, 1 }        
    sphere {  m*<0.10932611123758684,-4.834832497323224e-18,4.1616697129296>, 1 }
    sphere {  m*<9.056207785767104,1.0991639745120106e-18,-2.0480637529838495>, 1 }
    sphere {  m*<-4.631095730252666,8.164965809277259,-2.151946510623188>, 1}
    sphere { m*<-4.631095730252666,-8.164965809277259,-2.1519465106231914>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.10932611123758684,-4.834832497323224e-18,4.1616697129296>, <0.09681277307753526,-4.710574297700019e-18,1.1616954320169603>, 0.5 }
    cylinder { m*<9.056207785767104,1.0991639745120106e-18,-2.0480637529838495>, <0.09681277307753526,-4.710574297700019e-18,1.1616954320169603>, 0.5}
    cylinder { m*<-4.631095730252666,8.164965809277259,-2.151946510623188>, <0.09681277307753526,-4.710574297700019e-18,1.1616954320169603>, 0.5 }
    cylinder {  m*<-4.631095730252666,-8.164965809277259,-2.1519465106231914>, <0.09681277307753526,-4.710574297700019e-18,1.1616954320169603>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    