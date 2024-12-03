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
    sphere { m*<-0.8555753686210992,-0.16047334128988255,-1.4186818340091945>, 1 }        
    sphere {  m*<0.2754231836206567,0.28526522573758617,8.507130631934217>, 1 }
    sphere {  m*<4.784218175033322,0.04117364598819184,-4.180647899693353>, 1 }
    sphere {  m*<-2.5055266844723216,2.1683752850720897,-2.3428665032825284>, 1}
    sphere { m*<-2.2377394634344903,-2.7193166573318077,-2.153320218119958>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2754231836206567,0.28526522573758617,8.507130631934217>, <-0.8555753686210992,-0.16047334128988255,-1.4186818340091945>, 0.5 }
    cylinder { m*<4.784218175033322,0.04117364598819184,-4.180647899693353>, <-0.8555753686210992,-0.16047334128988255,-1.4186818340091945>, 0.5}
    cylinder { m*<-2.5055266844723216,2.1683752850720897,-2.3428665032825284>, <-0.8555753686210992,-0.16047334128988255,-1.4186818340091945>, 0.5 }
    cylinder {  m*<-2.2377394634344903,-2.7193166573318077,-2.153320218119958>, <-0.8555753686210992,-0.16047334128988255,-1.4186818340091945>, 0.5}

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
    sphere { m*<-0.8555753686210992,-0.16047334128988255,-1.4186818340091945>, 1 }        
    sphere {  m*<0.2754231836206567,0.28526522573758617,8.507130631934217>, 1 }
    sphere {  m*<4.784218175033322,0.04117364598819184,-4.180647899693353>, 1 }
    sphere {  m*<-2.5055266844723216,2.1683752850720897,-2.3428665032825284>, 1}
    sphere { m*<-2.2377394634344903,-2.7193166573318077,-2.153320218119958>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2754231836206567,0.28526522573758617,8.507130631934217>, <-0.8555753686210992,-0.16047334128988255,-1.4186818340091945>, 0.5 }
    cylinder { m*<4.784218175033322,0.04117364598819184,-4.180647899693353>, <-0.8555753686210992,-0.16047334128988255,-1.4186818340091945>, 0.5}
    cylinder { m*<-2.5055266844723216,2.1683752850720897,-2.3428665032825284>, <-0.8555753686210992,-0.16047334128988255,-1.4186818340091945>, 0.5 }
    cylinder {  m*<-2.2377394634344903,-2.7193166573318077,-2.153320218119958>, <-0.8555753686210992,-0.16047334128988255,-1.4186818340091945>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    