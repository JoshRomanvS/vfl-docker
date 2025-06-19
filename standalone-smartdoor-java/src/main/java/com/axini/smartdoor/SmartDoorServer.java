package com.axini.smartdoor;

// Copyright 2023 Axini B.V. https://www.axini.com, see: LICENSE.txt.

import java.net.InetSocketAddress;

import org.java_websocket.WebSocket;
import org.java_websocket.drafts.Draft;
import org.java_websocket.drafts.Draft_6455;
import org.java_websocket.handshake.ClientHandshake;
import org.java_websocket.server.WebSocketServer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

// SmartDoorServer: WebSocket server for the SmartDoor SUT.
public class SmartDoorServer extends WebSocketServer {
    private static Logger logger =
        LoggerFactory.getLogger(SmartDoorServer.class);

    private SmartDoor   door;
    private WebSocket   client_connection;

    public SmartDoorServer(InetSocketAddress address) {
        super(address);
        this.door = null;
        this.client_connection = null;
    }

    @Override
    public void onOpen(WebSocket conn, ClientHandshake handshake) {
        logger.info("Client connected");
        if (client_connection != null) {
            logger.error("Another client tried to connect to this WebSocketServer");
            conn.close();
        }
        else {
            client_connection = conn;
            door = new SmartDoor(this);
            logger.info("SmartDoor created");
        }
    }

    @Override
    public void onClose(WebSocket conn, int code, String reason, boolean remote) {
        logger.info("Client disconnected");
        client_connection = null;
        door = null;
    }

    @Override
    public void onMessage(WebSocket conn, String message) {
        logger.info("Received message: " + message);

        String[] arr = message.split(":");
        String action = arr[0];
        String param = (arr.length == 2) ? arr[1] : null;

        switch(action) {
        case "RESET":
            // We are ignoring any manufacturer param.
            reset_sut();
            break;
        default:
            door.handle_input(message);
        }
    }

    @Override
    public void onError(WebSocket conn, Exception ex) {
        logger.error("Error occurred: " + ex.getMessage());
    }

    @Override
    public void onStart() {
        System.out.println("Server started!");
        setConnectionLostTimeout(0);
        setConnectionLostTimeout(100);
    }

    public void send(String message) {
        logger.info("Sending message:  " + message);
        if (client_connection == null)
            logger.error("WebSocket connection is not yet initialised");
        else
            client_connection.send(message);
    }

    private void reset_sut() {
        logger.info("Resetting SUT");
        door = new SmartDoor(this);
        send("RESET_PERFORMED");
    }
}
