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
    sphere { m*<-1.201146608090824,-0.7834130164806552,-0.8252105643061587>, 1 }        
    sphere {  m*<0.242180938733094,-0.06544343151304768,9.044077656057135>, 1 }
    sphere {  m*<7.597532376733063,-0.15436370750740486,-5.5354156339882135>, 1 }
    sphere {  m*<-5.269738619874191,4.299524669163409,-2.9089651656102564>, 1}
    sphere { m*<-2.488079510487647,-3.418030878840059,-1.4583961442255042>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.242180938733094,-0.06544343151304768,9.044077656057135>, <-1.201146608090824,-0.7834130164806552,-0.8252105643061587>, 0.5 }
    cylinder { m*<7.597532376733063,-0.15436370750740486,-5.5354156339882135>, <-1.201146608090824,-0.7834130164806552,-0.8252105643061587>, 0.5}
    cylinder { m*<-5.269738619874191,4.299524669163409,-2.9089651656102564>, <-1.201146608090824,-0.7834130164806552,-0.8252105643061587>, 0.5 }
    cylinder {  m*<-2.488079510487647,-3.418030878840059,-1.4583961442255042>, <-1.201146608090824,-0.7834130164806552,-0.8252105643061587>, 0.5}

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
    sphere { m*<-1.201146608090824,-0.7834130164806552,-0.8252105643061587>, 1 }        
    sphere {  m*<0.242180938733094,-0.06544343151304768,9.044077656057135>, 1 }
    sphere {  m*<7.597532376733063,-0.15436370750740486,-5.5354156339882135>, 1 }
    sphere {  m*<-5.269738619874191,4.299524669163409,-2.9089651656102564>, 1}
    sphere { m*<-2.488079510487647,-3.418030878840059,-1.4583961442255042>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.242180938733094,-0.06544343151304768,9.044077656057135>, <-1.201146608090824,-0.7834130164806552,-0.8252105643061587>, 0.5 }
    cylinder { m*<7.597532376733063,-0.15436370750740486,-5.5354156339882135>, <-1.201146608090824,-0.7834130164806552,-0.8252105643061587>, 0.5}
    cylinder { m*<-5.269738619874191,4.299524669163409,-2.9089651656102564>, <-1.201146608090824,-0.7834130164806552,-0.8252105643061587>, 0.5 }
    cylinder {  m*<-2.488079510487647,-3.418030878840059,-1.4583961442255042>, <-1.201146608090824,-0.7834130164806552,-0.8252105643061587>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    